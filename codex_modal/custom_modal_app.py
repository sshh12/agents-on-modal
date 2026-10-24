"""Generic Modal app deployed dynamically for non-catalog Hugging Face weights.

The local wrapper supplies all model and hardware choices through a non-secret JSON
environment variable. This module intentionally contains no model-specific preset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from . import runtimes
from .custom_deploy import CUSTOM_APP_CONFIG_ENV


raw_config = os.environ.get(CUSTOM_APP_CONFIG_ENV)
if not raw_config:
    raise RuntimeError(
        f"{CUSTOM_APP_CONFIG_ENV} is required; deploy this module through codex-modal."
    )
config: dict[str, Any] = json.loads(raw_config)
runtime = runtimes.runtime_for(config.get("runtime"))

MODEL_MOUNT = "/codex-model-store"
MODEL_PATH = f"{MODEL_MOUNT}/model"
BASE_MOUNT = "/codex-base-model"
HF_RUNTIME_ENV = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}
REMOTE_CONFIG_ENV = {CUSTOM_APP_CONFIG_ENV: raw_config} | HF_RUNTIME_ENV


def _hf_secrets() -> list[modal.Secret]:
    token_env = config.get("hf_token_env")
    if not token_env:
        return []
    # Declare the secret unconditionally so the function's dependency COUNT is the
    # same at local deploy time and at remote container hydration. Modal binds a
    # container's object ids to code dependencies positionally, so gating this on
    # modal.is_local() (which is False inside the container) produced "Function
    # has N dependencies but container got N+1 object ids". The token value is
    # captured from the local env at deploy; the deployed secret supplies it in
    # the container, so the empty fallback here is only a structural placeholder.
    token = os.environ.get(str(token_env), "")
    if modal.is_local() and not token:
        raise RuntimeError(f"Environment variable {token_env!r} is empty or missing.")
    return [modal.Secret.from_dict({"HF_TOKEN": token})]


app = modal.App(name=str(config["app_name"]))
model_volume = modal.Volume.from_name(
    str(config["volume_name"]), create_if_missing=True
)
base_volume_name = config.get("base_volume")
base_volume = (
    modal.Volume.from_name(str(base_volume_name)) if base_volume_name else None
)

volume_mounts: dict[str, modal.Volume] = {MODEL_MOUNT: model_volume}
if base_volume is not None:
    volume_mounts[BASE_MOUNT] = base_volume

prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.35,<2")
    .env(REMOTE_CONFIG_ENV)
)
# The selected runtime owns image construction (SGLang image vs. a CUDA base
# that compiles llama.cpp). The env + extra wheels are layered on generically.
serving_image = runtime.build_serving_image(str(config["sglang_image"]), config)
# Extra wheels layered onto the serving image, installed last so an explicit
# pin wins over whatever the base image or the runtime resolved - e.g. a
# Transformers version new enough to recognize a very new model architecture.
_serving_pip = tuple(str(package) for package in (config.get("serving_pip") or ()))
if _serving_pip:
    serving_image = serving_image.uv_pip_install(*_serving_pip)
serving_image = serving_image.env(REMOTE_CONFIG_ENV)


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
    timeout=max(3_600, int(config["startup_timeout"])),
    max_containers=1,
    volumes=volume_mounts,
    secrets=_hf_secrets(),
)
def prepare_model() -> dict[str, Any]:
    """Materialize a checkpoint, reusing only content-identical base files."""

    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    source_info = api.model_info(
        repo_id=str(config["source_repo"]),
        revision=str(config["source_revision"]),
        files_metadata=True,
    )
    source_revision = str(source_info.sha)
    spec = runtime.download_spec(config)
    source_files = {
        str(sibling.rfilename): sibling for sibling in source_info.siblings
    }
    if spec.subdir:
        prefix = spec.subdir.strip("/") + "/"
        source_files = {
            name: sibling
            for name, sibling in source_files.items()
            if name.startswith(prefix)
        }
        if not source_files:
            raise RuntimeError(
                f"Repo subdirectory {spec.subdir!r} matched no files in "
                f"{config['source_repo']}@{source_revision}."
            )

    base_files: dict[str, Any] = {}
    base_root: Path | None = None
    if config.get("base_volume"):
        base_info = api.model_info(
            repo_id=str(config["base_repo"]),
            revision=str(config["base_revision"]),
            files_metadata=True,
        )
        base_files = {
            str(sibling.rfilename): sibling for sibling in base_info.siblings
        }
        base_relative = str(config["base_volume_path"]).lstrip("/")
        base_root = Path(BASE_MOUNT, base_relative)
        if not (base_root / "config.json").is_file():
            raise RuntimeError(
                f"Base model path {base_root} does not contain config.json."
            )

    target = Path(MODEL_PATH)
    if target.is_symlink() or target.is_file():
        target.unlink()
    target.mkdir(parents=True, exist_ok=True)

    download_names: list[str] = []
    reused: list[tuple[str, Path]] = []
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

    pending_downloads: list[str] = []
    for filename in download_names:
        relative = _safe_relative_path(filename)
        destination = target.joinpath(*relative.parts)
        expected_size = getattr(source_files[filename], "size", None)
        complete = destination.is_file() and (
            expected_size is None or destination.stat().st_size == int(expected_size)
        )
        if not complete:
            if destination.exists() or destination.is_symlink():
                if destination.is_dir():
                    raise RuntimeError(
                        f"Expected a model file but found a directory: {destination}"
                    )
                destination.unlink()
            pending_downloads.append(filename)

    if pending_downloads:
        print(
            f"Downloading {len(pending_downloads)} missing/changed file(s) from "
            f"{config['source_repo']}@{source_revision}; reusing {len(reused)} "
            "content-identical base file(s).",
            flush=True,
        )
        snapshot_download(
            repo_id=str(config["source_repo"]),
            revision=source_revision,
            local_dir=str(target),
            allow_patterns=pending_downloads,
            token=token,
            max_workers=8,
        )

    for filename, base_candidate in reused:
        relative = _safe_relative_path(filename)
        destination = target.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if destination.is_dir():
                raise RuntimeError(
                    f"Expected a model file but found a directory: {destination}"
                )
            destination.unlink()
        destination.symlink_to(base_candidate)

    incomplete: list[str] = []
    for filename, source_file in source_files.items():
        relative = _safe_relative_path(filename)
        destination = target.joinpath(*relative.parts)
        expected_size = getattr(source_file, "size", None)
        if not destination.is_file() or (
            expected_size is not None
            and destination.stat().st_size != int(expected_size)
        ):
            incomplete.append(filename)
    if incomplete:
        preview = ", ".join(incomplete[:10])
        suffix = "..." if len(incomplete) > 10 else ""
        raise RuntimeError(
            f"Prepared repository is incomplete: {len(incomplete)} file(s) missing or "
            f"wrong-sized ({preview}{suffix})."
        )
    if spec.needs_config_json:
        if not (target / "config.json").is_file():
            raise RuntimeError(
                f"Prepared repository {config['source_repo']} has no root config.json."
            )
    else:
        search_root = target / spec.subdir if spec.subdir else target
        if not any(search_root.rglob("*.gguf")):
            raise RuntimeError(
                f"Prepared repository {config['source_repo']} has no .gguf files under "
                f"{search_root}."
            )
    manifest = {
        "source_repo": config["source_repo"],
        "source_revision": source_revision,
        "downloaded_files": len(pending_downloads),
        "reused_files": len(reused),
    }
    (Path(MODEL_MOUNT) / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    model_volume.commit()
    print(f"Prepared model: {json.dumps(manifest, sort_keys=True)}", flush=True)
    return manifest


compute_region: str | list[str] | None
if bool(config["colocate_compute"]):
    compute_region = str(config["routing_region"])
elif config["compute_regions"]:
    compute_region = [str(region) for region in config["compute_regions"]]
else:
    compute_region = None

@app.server(
    image=serving_image,
    gpu=str(config["gpu"]),
    cpu=int(config["cpu"]),
    memory=int(config["memory"]),
    min_containers=0,
    max_containers=int(config["max_containers"]),
    scaledown_window=int(config["scaledown_window"]),
    port=int(config["port"]),
    routing_region=str(config["routing_region"]),
    compute_region=compute_region,
    unauthenticated=False,
    exit_grace_period=25,
    startup_timeout=int(config["startup_timeout"]),
    target_concurrency=int(config["target_inputs"]),
    volumes=volume_mounts,
)
class Server:
    @modal.enter()
    def startup(self) -> None:
        self.handle = runtime.start(config, MODEL_PATH, int(config["port"]))
        print(runtime.ready_message(config), flush=True)

    @modal.exit()
    def stop(self) -> None:
        if hasattr(self, "handle"):
            self.handle.stop()
