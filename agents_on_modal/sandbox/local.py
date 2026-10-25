"""The ``local`` sandbox backend: run the harness directly on the host, no container.

This is the "work on this repo itself" flow. The harness process, the runner, and every exec share
the host OS, the host network, and the user's own account -- there is no isolation and no proxy. The
model endpoint is reached directly through the gateway with the user's real token.

State for each local sandbox lives under ``<AOM_HOME>/local/<id>/``::

    <AOM_HOME>/local/<id>/meta.json     rediscovery record (list()/get() read this)
    <AOM_HOME>/local/<id>/home/         the sandbox $HOME
    <AOM_HOME>/local/<id>/home/runs/    run directories (runs_root)

``create`` persists ``meta.json`` so a sandbox created in one process can be reattached from another
with ``get(id)`` or found with ``list()``. ``stop`` removes the state directory and is idempotent.

Everything here is pure host code (subprocess + pathlib) and works on Windows and POSIX alike.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ..config import home as _aom_home
from ..harness.base import ModelAccess, SandboxProfile
from .base import ExecResult, Sandbox, SandboxError, new_id
from .instructions import build_instructions, load_skills

#: Win32 process-creation flags that fully detach a child (survives the parent).
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008


def _state_root() -> Path:
    """``<AOM_HOME>/local`` -- the directory that holds one subdir per local sandbox."""

    root = _aom_home() / "local"
    root.mkdir(parents=True, exist_ok=True)
    return root


class LocalSandbox(Sandbox):
    """A sandbox that is just the host machine. Cheap handle; state is files on disk."""

    kind: ClassVar[str] = "local"

    def __init__(self, meta: Mapping[str, Any]) -> None:
        self._meta: dict[str, Any] = dict(meta)
        self.id = str(meta["id"])
        self.workdir = str(meta["workdir"])
        self.home = str(meta["home"])
        self.runs_root = str(meta["runs_root"])
        self.posix = bool(meta.get("posix", os.name != "nt"))
        #: aom is importable in this very interpreter, so reuse it for the runner.
        self.python = (sys.executable,)

    # ---- lifecycle --------------------------------------------------------------------------
    @classmethod
    def create(cls, *, workdir: str | None = None, **_: Any) -> "LocalSandbox":
        workdir = os.path.abspath(workdir or os.getcwd())
        sandbox_id = new_id("local-")
        state_dir = _state_root() / sandbox_id
        home_dir = state_dir / "home"
        runs_root = home_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": sandbox_id,
            "kind": cls.kind,
            "workdir": workdir,
            "home": str(home_dir),
            "runs_root": str(runs_root),
            "posix": os.name != "nt",
            "created_at": time.time(),
        }
        _write_json(state_dir / "meta.json", meta)
        return cls(meta)

    @classmethod
    def list(cls, **filters: Any) -> list["LocalSandbox"]:
        wanted_workdir = filters.get("workdir")
        if wanted_workdir is not None:
            wanted_workdir = os.path.abspath(wanted_workdir)
        boxes: list[LocalSandbox] = []
        for meta_path in _state_root().glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(meta, dict) or "id" not in meta:
                continue
            if wanted_workdir is not None and os.path.abspath(str(meta.get("workdir", ""))) != wanted_workdir:
                continue
            boxes.append(cls(meta))
        boxes.sort(key=lambda b: (b._meta.get("created_at", 0.0), b.id))
        return boxes

    @classmethod
    def get(cls, sandbox_id: str) -> "LocalSandbox":
        meta_path = _state_root() / sandbox_id / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SandboxError(f"No local sandbox {sandbox_id!r} ({meta_path} unreadable: {error}).") from error
        return cls(meta)

    def stop(self) -> None:
        shutil.rmtree(_state_root() / self.id, ignore_errors=True)

    def labels(self) -> dict[str, str]:
        return {"kind": self.kind, "workdir": self.workdir}

    # ---- primitives -------------------------------------------------------------------------
    def exec(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None,
             timeout: float | None = 120, root: bool = False) -> ExecResult:
        # root is ignored: the local sandbox always runs as the user who launched aom.
        try:
            proc = subprocess.run(
                list(argv), env=self._env(env), cwd=cwd or self.workdir, timeout=timeout,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as error:
            raise SandboxError(f"command {list(argv)!r} timed out after {timeout}s") from error
        except OSError as error:
            raise SandboxError(f"could not run {list(argv)!r}: {error}") from error
        return ExecResult(proc.returncode, proc.stdout or "", proc.stderr or "")

    def spawn(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None) -> None:
        kwargs: dict[str, Any] = dict(
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=cwd or self.workdir, env=self._env(env),
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen(list(argv), **kwargs)  # detached; we drop the handle on purpose
        except OSError as error:
            raise SandboxError(f"could not spawn {list(argv)!r}: {error}") from error

    def write_file(self, path: str, data: str | bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write bytes either way so no newline translation happens on Windows: the content must
        # round-trip exactly through read_file (which reads raw bytes).
        target.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))

    def read_file(self, path: str, offset: int = 0) -> bytes:
        # open(...) raises FileNotFoundError when missing, which is exactly the contract.
        with open(path, "rb") as stream:
            if offset:
                stream.seek(offset)
            return stream.read()

    def attach(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None) -> int:
        # Inherit the caller's real stdio so an interactive TUI drives the terminal.
        try:
            proc = subprocess.run(list(argv), env=self._env(env), cwd=cwd or self.workdir)
        except OSError as error:
            raise SandboxError(f"could not run {list(argv)!r}: {error}") from error
        return proc.returncode

    def model_access(self, endpoint: Any) -> ModelAccess:
        # Local shares the host network, so the harness talks to the gateway directly.
        return ModelAccess.from_endpoint(endpoint)

    def profile(self) -> SandboxProfile:
        return SandboxProfile(
            kind="local",
            instructions=build_instructions(
                "local", workdir=self.workdir, home=self.home,
                shared_dir=None, network="host", proxy=False,
            ),
            posix=self.posix,
            skills=load_skills(),
            mcp_servers={},
            env={},
        )

    # ---- helpers ----------------------------------------------------------------------------
    def _env(self, env: Mapping[str, str] | None) -> dict[str, str] | None:
        """Merge ``env`` over the host environment; ``None`` inherits os.environ unchanged."""

        if env is None:
            return None
        return {**os.environ, **env}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = ["LocalSandbox"]
