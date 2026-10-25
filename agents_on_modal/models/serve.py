"""The Modal app module for one model, deployed with ``modal deploy -m agents_on_modal.models.serve``.

The model spec arrives as the non-secret env var ``AOM_MODEL_JSON`` (``Model.to_json()``),
and the deployed app name as ``AOM_APP_NAME`` (``deploy.py`` computes a DNS-safe, possibly
shortened name and uses the same value for ``modal deploy --name``). The module builds, at
import time (identically on the host at deploy and inside the container at hydration):

  * a CPU ``prepare`` function that downloads and verifies the weights to a persistent
    Volume before any GPU is requested (cost-critical), then commits; and
  * an ``@app.server`` class whose ``@modal.enter`` launches the engine subprocess on
    ``127.0.0.1:8001`` and runs the gateway on the server port ``8000``. If the engine
    dies the container exits so Modal replaces it.

``modal`` is imported at module top because this module is only ever run by ``modal deploy``
or inside a Modal container; nothing here *calls* Modal at import, so a bare
``import agents_on_modal.models.serve`` succeeds off-host.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from .engines import get as get_engine

AOM_MODEL_JSON_ENV = "AOM_MODEL_JSON"
AOM_APP_NAME_ENV = "AOM_APP_NAME"

# Mount points and ports inside the GPU container.
MODEL_MOUNT = "/aom-weights"
MODEL_PATH = f"{MODEL_MOUNT}/model"
BASE_MOUNT = "/aom-base"
ENGINE_PORT = 8001  # the engine binds loopback here
SERVER_PORT = 8000  # the gateway binds 0.0.0.0 here; Modal routes to it

# Hugging Face download environment (A13): fast xet transfer, no telemetry.
HF_PREPARE_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

# Used only for a bare offline import (introspection, unit tests). A real deploy always
# sets AOM_MODEL_JSON, so this stub never reaches a container.
_STUB_MODEL_JSON: dict[str, Any] = {
    "name": "aom-import-stub",
    "repo": "stub/stub",
    "engine": "sglang",
    "gpu": "H100",
    "context_window": 8192,
    "max_output_tokens": 1024,
    "image": "lmsysorg/sglang:v0.5.22-cu130",
    "region": "us-west",
    "scaling": {},
}


def _load_model_json() -> dict[str, Any]:
    raw = os.environ.get(AOM_MODEL_JSON_ENV)
    if not raw:
        return dict(_STUB_MODEL_JSON)
    return json.loads(raw)


model_json: dict[str, Any] = _load_model_json()
engine = get_engine(str(model_json["engine"]))

app_name = os.environ.get(AOM_APP_NAME_ENV) or f"aom-{str(model_json['name']).replace('.', '-')}"
# The weights Volume name is derived from the model name (not the possibly-shortened app
# name) so it is stable across deploys; mirrors Model.volume_name.
volume_name = f"aom-{str(model_json['name']).replace('.', '-')}-weights"

_scaling = dict(model_json.get("scaling") or {})
_startup_timeout = int(model_json.get("startup_timeout", 3600))

app = modal.App(name=app_name)
model_volume = modal.Volume.from_name(volume_name, version=2, create_if_missing=True)
_volumes = {MODEL_MOUNT: model_volume}

# Optional base-volume reuse (content-identical files are symlinked, not re-downloaded).
# The current Model contract carries no base_* fields, so this stays inert unless a future
# catalog supplies them; the plumbing is preserved so it can be re-enabled without changes.
_base_volume_name = model_json.get("base_volume")
base_volume = modal.Volume.from_name(str(_base_volume_name), version=2) if _base_volume_name else None
if base_volume is not None:
    _volumes[BASE_MOUNT] = base_volume

prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.35,<2")
    .env(HF_PREPARE_ENV)
)

# The serving image is the engine's image plus the aom package source baked in (copy=True),
# so the gateway is importable in the container.
serving_image = engine.image(model_json).add_local_python_source("agents_on_modal", copy=True)


def _hf_secrets() -> list["modal.Secret"]:
    """Declare the HF token secret unconditionally (A1: a module-level Modal object must
    be built identically locally and remotely, or the container gets an object-id mismatch).
    The value is captured from the local env at deploy; the empty fallback is a structural
    placeholder in the container, and the missing-token *guard* is the only is_local() branch.
    """

    token_env = model_json.get("hf_token_env")
    if not token_env:
        return []
    token = os.environ.get(str(token_env), "")
    if modal.is_local() and not token:
        raise RuntimeError(f"Environment variable {token_env!r} is empty or missing.")
    return [modal.Secret.from_dict({"HF_TOKEN": token})]


def _safe_relative_path(filename: str) -> PurePosixPath:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"Hugging Face returned an unsafe repo filename: {filename!r}")
    return path


def _metadata_identity(sibling: Any) -> tuple[str, str] | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs:
        sha256 = getattr(lfs, "sha256", None)
        if sha256:
            return "sha256", str(sha256)
    blob_id = getattr(sibling, "blob_id", None)
    if blob_id:
        return "git-blob", str(blob_id)
    return None


@app.function(
    image=prepare_image,
    cpu=4,
    memory=16_384,
    timeout=max(3_600, _startup_timeout),
    max_containers=1,
    volumes=_volumes,
    secrets=_hf_secrets(),
)
def prepare() -> dict[str, Any]:
    """Materialize the checkpoint to the Volume on CPU: resumable, size-verified, committed.

    Ported from the old ``custom_modal_app.prepare_model`` (A11/A12): pin the revision to
    ``model_info().sha``, re-download wrong-sized files, verify every file's size, write a
    manifest, then commit. Large downloads never bill GPU time.
    """

    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    repo = str(model_json["repo"])
    revision = str(model_json.get("revision") or "main")
    spec = engine.download_spec(model_json)

    api = HfApi(token=token)
    info = api.model_info(repo_id=repo, revision=revision, files_metadata=True)
    source_revision = str(info.sha)
    source_files = {str(s.rfilename): s for s in info.siblings}
    if spec.subdir:
        prefix = spec.subdir.strip("/") + "/"
        source_files = {n: s for n, s in source_files.items() if n.startswith(prefix)}
        if not source_files:
            raise RuntimeError(
                f"Repo subdirectory {spec.subdir!r} matched no files in {repo}@{source_revision}."
            )

    # Optional base-volume reuse: symlink a base file only when LFS sha256 (or git blob id)
    # AND size match, and the base file exists at the right size.
    base_files: dict[str, Any] = {}
    base_root: Path | None = None
    if _base_volume_name and model_json.get("base_repo"):
        base_info = api.model_info(
            repo_id=str(model_json["base_repo"]),
            revision=str(model_json.get("base_revision") or "main"),
            files_metadata=True,
        )
        base_files = {str(s.rfilename): s for s in base_info.siblings}
        base_root = Path(BASE_MOUNT, str(model_json.get("base_volume_path", "")).lstrip("/"))

    target = Path(MODEL_PATH)
    if target.is_symlink() or target.is_file():
        target.unlink()
    target.mkdir(parents=True, exist_ok=True)

    reused: list[tuple[str, Path]] = []
    download_names: list[str] = []
    for filename, source_file in source_files.items():
        relative = _safe_relative_path(filename)
        base_file = base_files.get(filename)
        base_candidate = base_root.joinpath(*relative.parts) if base_root else None
        same_content = (
            base_file is not None
            and _metadata_identity(source_file) is not None
            and _metadata_identity(source_file) == _metadata_identity(base_file)
            and getattr(source_file, "size", None) == getattr(base_file, "size", None)
        )
        base_size = getattr(base_file, "size", None) if base_file is not None else None
        base_is_complete = (
            base_candidate is not None
            and base_candidate.is_file()
            and (base_size is None or base_candidate.stat().st_size == int(base_size))
        )
        if same_content and base_is_complete:
            reused.append((filename, base_candidate))
        else:
            download_names.append(filename)

    # Resume: only fetch files that are missing or the wrong size.
    pending: list[str] = []
    for filename in download_names:
        relative = _safe_relative_path(filename)
        destination = target.joinpath(*relative.parts)
        expected = getattr(source_files[filename], "size", None)
        complete = destination.is_file() and (
            expected is None or destination.stat().st_size == int(expected)
        )
        if not complete:
            if destination.is_dir():
                raise RuntimeError(f"Expected a model file but found a directory: {destination}")
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            pending.append(filename)

    if pending:
        print(
            f"Downloading {len(pending)} file(s) from {repo}@{source_revision}; "
            f"reusing {len(reused)} base file(s).",
            flush=True,
        )
        snapshot_download(
            repo_id=repo,
            revision=source_revision,
            local_dir=str(target),
            allow_patterns=pending,
            token=token,
            max_workers=8,
        )

    for filename, base_candidate in reused:
        relative = _safe_relative_path(filename)
        destination = target.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(base_candidate)

    # Verify completeness before committing.
    incomplete: list[str] = []
    for filename, source_file in source_files.items():
        relative = _safe_relative_path(filename)
        destination = target.joinpath(*relative.parts)
        expected = getattr(source_file, "size", None)
        if not destination.is_file() or (
            expected is not None and destination.stat().st_size != int(expected)
        ):
            incomplete.append(filename)
    if incomplete:
        preview = ", ".join(incomplete[:10]) + ("..." if len(incomplete) > 10 else "")
        raise RuntimeError(f"Prepared repo is incomplete: {len(incomplete)} file(s) ({preview}).")

    if spec.needs_config_json:
        if not (target / "config.json").is_file():
            raise RuntimeError(f"Prepared repo {repo} has no root config.json.")
    else:
        search_root = target / spec.subdir if spec.subdir else target
        if not any(search_root.rglob("*.gguf")):
            raise RuntimeError(f"Prepared repo {repo} has no .gguf files under {search_root}.")

    manifest = {
        "repo": repo,
        "revision": source_revision,
        "downloaded_files": len(pending),
        "reused_files": len(reused),
    }
    (Path(MODEL_MOUNT) / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    model_volume.commit()
    print(f"Prepared model: {json.dumps(manifest, sort_keys=True)}", flush=True)
    return manifest


@app.server(
    image=serving_image,
    gpu=str(model_json["gpu"]),
    cpu=int(model_json.get("cpu", 8)),
    memory=int(model_json.get("memory", 98_304)),
    min_containers=int(_scaling.get("min_containers", 0)),
    max_containers=int(_scaling.get("max_containers", 1)),
    buffer_containers=int(_scaling.get("buffer_containers", 0)),
    scaledown_window=int(_scaling.get("scaledown_window", 900)),
    target_concurrency=int(round(float(_scaling.get("target_concurrency", 32.0)))),
    port=SERVER_PORT,
    routing_region=str(model_json.get("region", "us-west")),
    unauthenticated=False,
    exit_grace_period=25,
    startup_timeout=_startup_timeout,
    volumes=_volumes,
)
class Server:
    @modal.enter()
    def startup(self) -> None:
        argv = engine.launch_argv(model_json, MODEL_PATH, ENGINE_PORT)
        env = {**os.environ, **engine.env(model_json)}
        print(f"[aom] launching engine: {' '.join(argv)}", flush=True)
        self._engine = subprocess.Popen(argv, env=env)
        # The container exits if the engine dies, so Modal replaces it.
        threading.Thread(target=self._watch_engine, daemon=True).start()
        # The gateway binds the server port and holds requests until the engine is healthy.
        threading.Thread(target=self._run_gateway, daemon=True).start()

    def _watch_engine(self) -> None:
        code = self._engine.wait()
        print(f"[aom] engine exited with {code}; stopping container", flush=True)
        os._exit(1 if code else 0)

    def _run_gateway(self) -> None:
        from ..gateway import server as gateway

        try:
            gateway.run(
                model_json,
                f"http://127.0.0.1:{ENGINE_PORT}",
                host="0.0.0.0",
                port=SERVER_PORT,
            )
        except BaseException as exc:  # noqa: BLE001 - last-resort container exit
            print(f"[aom] gateway crashed: {exc!r}", flush=True)
        os._exit(1)

    @modal.exit()
    def shutdown(self) -> None:
        proc = getattr(self, "_engine", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                proc.kill()
