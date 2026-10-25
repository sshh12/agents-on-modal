"""The sandbox contract plus everything that is identical across sandbox kinds.

Concrete sandboxes implement a handful of primitives (``exec``, ``spawn``, ``write_file``,
``read_file``, ``attach``, ``model_access``, ``profile``, lifecycle). Runs, traces, harness setup and
interactive sessions are built on those primitives here, so every sandbox behaves the same.

Run directory layout (inside the sandbox's filesystem)::

    <runs_root>/<run_id>/spec.json     what to run (written by ``start``)
    <runs_root>/<run_id>/trace.jsonl   envelopes (see ``agents_on_modal.traces``), appended live
    <runs_root>/<run_id>/status.json   runner state, rewritten atomically
    <runs_root>/<run_id>/harness/      the run's private harness home (transcripts, sessions)
"""

from __future__ import annotations

import dataclasses
import json
import posixpath
import secrets
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from .. import harness as harness_registry
from ..config import AomError
from ..harness.base import Harness, HarnessSetup, ModelAccess, SandboxProfile
from ..traces import Event, runner_events

TERMINAL_STATES = ("done", "timeout", "failed", "cancelled")


class SandboxError(AomError):
    """A sandbox could not be created, driven, or torn down."""


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str

    def check(self, what: str = "command") -> "ExecResult":
        if self.returncode != 0:
            raise SandboxError(f"{what} failed with exit code {self.returncode}: {self.stderr.strip() or self.stdout.strip()}")
        return self


def new_id(prefix: str = "") -> str:
    """Sortable, DNS-safe id: ``<prefix><yyyymmddhhmmss>-<hex6>``."""

    return f"{prefix}{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{secrets.token_hex(3)}"


class Sandbox(ABC):
    """One place harnesses run. Instances are cheap handles; state lives in the sandbox itself."""

    kind: ClassVar[str] = ""

    #: Stable id (container name, Modal sandbox id, or local dir name).
    id: str
    #: Agent working directory (``/work`` in containers, the chosen dir locally).
    workdir: str
    #: Agent home directory.
    home: str
    #: Root of run directories.
    runs_root: str
    #: True when paths inside the sandbox are POSIX.
    posix: bool = True
    #: argv prefix that runs Python with ``agents_on_modal`` importable inside the sandbox.
    python: tuple[str, ...] = ("python3",)

    # ---- lifecycle (per kind) ---------------------------------------------------------------
    @classmethod
    @abstractmethod
    def create(cls, **options: Any) -> "Sandbox":
        """Create and start a new sandbox."""

    @classmethod
    @abstractmethod
    def list(cls, **filters: Any) -> list["Sandbox"]:
        """Live sandboxes of this kind created by aom (optionally filtered)."""

    @classmethod
    @abstractmethod
    def get(cls, sandbox_id: str) -> "Sandbox":
        """Reattach to an existing sandbox by id."""

    @abstractmethod
    def stop(self) -> None:
        """Destroy the sandbox and everything it owns (idempotent)."""

    def labels(self) -> dict[str, str]:
        """Metadata recorded at creation (tags / labels / meta.json)."""

        return {}

    # ---- primitives (per kind) --------------------------------------------------------------
    @abstractmethod
    def exec(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None,
             timeout: float | None = 120, root: bool = False) -> ExecResult:
        """Run a command to completion as the agent user (``root=True`` for setup tasks)."""

    @abstractmethod
    def spawn(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None) -> None:
        """Start a detached background process as the agent user; returns immediately."""

    @abstractmethod
    def write_file(self, path: str, data: str | bytes) -> None:
        """Write a file (creating parents), owned by the agent user."""

    @abstractmethod
    def read_file(self, path: str, offset: int = 0) -> bytes:
        """Bytes from ``offset`` to EOF. Raises ``FileNotFoundError`` when missing."""

    @abstractmethod
    def attach(self, argv: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None) -> int:
        """Run an interactive command on the caller's terminal; returns its exit code."""

    @abstractmethod
    def model_access(self, endpoint: Any) -> ModelAccess:
        """How the model endpoint is reached from inside this sandbox."""

    @abstractmethod
    def profile(self) -> SandboxProfile:
        """Instructions, MCP servers, skills and env this sandbox gives every harness."""

    # ---- shared behaviour -------------------------------------------------------------------
    def join(self, *parts: str) -> str:
        if self.posix:
            return posixpath.join(*parts)
        import ntpath

        return ntpath.join(*parts)

    def setup_harness(self, harness: str | Harness, endpoint: Any, *, home: str | None = None,
                      effort: str | None = None) -> tuple[Harness, HarnessSetup]:
        """Prepare a harness home inside the sandbox and write its files."""

        h = harness_registry.get(harness)
        access = self.model_access(endpoint)
        if effort:
            access = dataclasses.replace(access, default_effort=effort)
        home = home or self.join(self.home, ".aom", h.name)
        setup = h.prepare(access, self.profile(), home=home, workdir=self.workdir)
        for path, content in setup.files.items():
            self.write_file(path, content)
        return h, setup

    def start(self, harness: str | Harness, endpoint: Any, prompt: str, *, goal: bool = False,
              timeout: float | None = None, effort: str | None = None, run_id: str | None = None,
              meta: Mapping[str, Any] | None = None) -> "Run":
        """Start a headless run in the background and return its handle immediately."""

        run_id = run_id or new_id("run-")
        run_dir = self.join(self.runs_root, run_id)
        h, setup = self.setup_harness(harness, endpoint, home=self.join(run_dir, "harness"), effort=effort)
        spec = {
            "run_id": run_id,
            "harness": h.name,
            "model": endpoint.model.name,
            "prompt": prompt,
            "goal": goal,
            "created_at": time.time(),
            "deadline": time.time() + timeout if timeout else None,
            "setup": setup.to_json(),
            "sandbox": {"kind": self.kind, "id": self.id},
            "meta": dict(meta or {}),
        }
        self.write_file(self.join(run_dir, "spec.json"), json.dumps(spec, indent=2))
        self.spawn([*self.python, "-m", "agents_on_modal.runner", run_dir], cwd=setup.workdir)
        return Run(self, run_id)

    def runs(self) -> list["Run"]:
        """Every run recorded in this sandbox, oldest first."""

        listing = self.exec(
            [*self.python, "-c", _LIST_RUNS, self.runs_root], timeout=60
        )
        ids = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
        return [Run(self, run_id) for run_id in sorted(ids)]

    def run(self, run_id: str) -> "Run":
        return Run(self, run_id)

    def interactive(self, harness: str | Harness, endpoint: Any, *, effort: str | None = None) -> int:
        """Open the harness TUI in this sandbox on the caller's terminal."""

        h, setup = self.setup_harness(harness, endpoint, effort=effort)
        return self.attach(h.interactive_argv(setup), env=setup.env, cwd=setup.workdir)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}>"


_LIST_RUNS = (
    "import os,sys\n"
    "root=sys.argv[1]\n"
    "print('\\n'.join(n for n in (os.listdir(root) if os.path.isdir(root) else [])"
    " if os.path.isfile(os.path.join(root,n,'spec.json'))))"
)


class Run:
    """A handle on one headless run. Everything is read from the sandbox, so handles are
    reconstructible from ``(sandbox, run_id)`` alone, from any process."""

    def __init__(self, sandbox: Sandbox, run_id: str) -> None:
        self.sandbox = sandbox
        self.run_id = run_id
        self.dir = sandbox.join(sandbox.runs_root, run_id)
        self._spec: dict[str, Any] | None = None
        self._harness: Harness | None = None

    @property
    def spec(self) -> dict[str, Any]:
        if self._spec is None:
            self._spec = json.loads(self.sandbox.read_file(self.sandbox.join(self.dir, "spec.json")))
        return self._spec

    @property
    def harness(self) -> Harness:
        if self._harness is None:
            self._harness = harness_registry.get(self.spec["harness"])
        return self._harness

    def status(self) -> dict[str, Any]:
        """Runner status (``state`` in starting|running|done|timeout|failed|cancelled)."""

        try:
            return json.loads(self.sandbox.read_file(self.sandbox.join(self.dir, "status.json")))
        except (FileNotFoundError, ValueError):
            return {"state": "starting"}

    def done(self) -> bool:
        return self.status().get("state") in TERMINAL_STATES

    def read_trace(self, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Complete envelopes after byte ``offset`` and the offset to continue from."""

        try:
            chunk = self.sandbox.read_file(self.sandbox.join(self.dir, "trace.jsonl"), offset)
        except FileNotFoundError:
            return [], offset
        end = chunk.rfind(b"\n")
        if end < 0:
            return [], offset
        envelopes = []
        for line in chunk[: end + 1].splitlines():
            if line.strip():
                try:
                    envelopes.append(json.loads(line))
                except ValueError:
                    continue
        return envelopes, offset + end + 1

    def parse(self, envelope: dict[str, Any]) -> list[Event]:
        if envelope.get("src") == "runner":
            return runner_events(envelope)
        return self.harness.parse(envelope)

    def events(self, *, follow: bool = False, offset: int = 0, poll: float = 1.0) -> Iterator[Event]:
        """Normalized events; with ``follow`` keep tailing until the run reaches a terminal state."""

        while True:
            envelopes, offset = self.read_trace(offset)
            for envelope in envelopes:
                yield from self.parse(envelope)
            if not follow:
                return
            if not envelopes and self.done():
                envelopes, offset = self.read_trace(offset)
                for envelope in envelopes:
                    yield from self.parse(envelope)
                return
            if not envelopes:
                time.sleep(poll)

    def wait(self, timeout: float | None = None, poll: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            status = self.status()
            if status.get("state") in TERMINAL_STATES:
                return status
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"run {self.run_id} still {status.get('state')} after {timeout}s")
            time.sleep(poll)

    def cancel(self) -> None:
        """Ask the runner to stop (it terminates the harness and records ``cancelled``)."""

        self.sandbox.write_file(self.sandbox.join(self.dir, "cancel"), "cancel\n")

    def __repr__(self) -> str:
        return f"<Run {self.run_id} in {self.sandbox.kind}:{self.sandbox.id}>"
