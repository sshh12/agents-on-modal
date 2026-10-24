"""Validated configuration for a generic, self-managed Modal SGLang app."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from . import runtimes
from .options import ResolvedModel, WrapperOptions


CUSTOM_APP_CONFIG_ENV = "CODEX_MODAL_CUSTOM_APP_CONFIG"
CUSTOM_APP_MODULE = "codex_modal.custom_modal_app"
DEFAULT_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.16-cu130"
DEFAULT_AUTOINFERENCE_UTILS = "0.2.2"
MODAL_IMAGE_BUILDER_VERSION = "2025.06"
DEFAULT_CPU = 8
DEFAULT_MEMORY = 98_304
DEFAULT_SCALEDOWN_WINDOW = 300
DEFAULT_TARGET_INPUTS = 16
DEFAULT_MAX_CONTAINERS = 1
DEFAULT_PORT = 8000
MAX_DNS_LABEL_LENGTH = 63

GPU_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_-]{0,31})(?::([1-8]))?$")
VOLUME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62})$")
SGLANG_KEY_PATTERN = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
RESERVED_SGLANG_ARGS = {
    "--host",
    "--model-path",
    "--port",
    "--served-model-name",
    "--tp",
}


@dataclass(frozen=True)
class CustomDeployment:
    app_name: str
    volume_name: str
    served_model: str
    source_repo: str
    source_revision: str
    base_repo: str | None
    base_revision: str | None
    base_volume: str | None
    base_volume_path: str | None
    gpu: str
    tensor_parallel_size: int
    # The serving engine ("sglang" | "llamacpp") and its base registry image.
    # ``sglang_image`` keeps its historical name for config/catalog
    # compatibility but is really the serving base image for whichever runtime
    # is selected (for llama.cpp this is a CUDA devel base that then compiles
    # the engine from source).
    runtime: str
    sglang_image: str
    autoinference_utils_version: str
    server_args: dict[str, str]
    serving_pip: tuple[str, ...]
    # Restrict the repo download to one subdirectory (e.g. a single GGUF quant).
    download_subdir: str | None
    cpu: int
    memory: int
    scaledown_window: int
    target_inputs: int
    max_containers: int
    startup_timeout: int
    routing_region: str
    compute_regions: tuple[str, ...]
    colocate_compute: bool
    hf_token_env: str | None
    port: int = DEFAULT_PORT

    def json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    def deployment_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment[CUSTOM_APP_CONFIG_ENV] = self.json()
        environment["MODAL_IMAGE_BUILDER_VERSION"] = MODAL_IMAGE_BUILDER_VERSION
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        return environment


def custom_volume_name(app_name: str) -> str:
    digest = hashlib.sha256(app_name.encode("utf-8")).hexdigest()[:10]
    name = f"cm-{app_name[:47].rstrip('-')}-{digest}"
    if not VOLUME_PATTERN.fullmatch(name):
        raise ValueError(f"Could not derive a safe Modal Volume name from {app_name!r}.")
    return name


def custom_app_name(requested_name: str, workspace_slug: str) -> str:
    """Fit a Modal Server app name inside its full DNS label."""

    fixed_length = len(workspace_slug) + len("--") + len("-server")
    maximum = MAX_DNS_LABEL_LENGTH - fixed_length
    if maximum < 10:
        raise ValueError(
            f"Modal workspace slug {workspace_slug!r} leaves no safe Server app-name space."
        )
    if len(requested_name) <= maximum:
        return requested_name
    digest = hashlib.sha256(requested_name.encode("utf-8")).hexdigest()[:8]
    prefix = requested_name[: maximum - len(digest) - 1].rstrip("-")
    shortened = f"{prefix}-{digest}"
    if not prefix or len(shortened) > maximum:
        raise ValueError(f"Could not derive a DNS-safe Modal app name from {requested_name!r}.")
    return shortened


def custom_app_base_url(
    workspace_slug: str, app_name: str, routing_region: str
) -> str:
    label = f"{workspace_slug}--{app_name}-server"
    if len(label) > MAX_DNS_LABEL_LENGTH:
        raise ValueError(
            f"Modal Server DNS label is {len(label)} characters; the limit is "
            f"{MAX_DNS_LABEL_LENGTH}. Shorten the app name first."
        )
    return (
        f"https://{label}."
        f"{routing_region}.modal.direct/v1"
    )


def _repo_id(value: str, label: str) -> str:
    if not value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{label} is empty or invalid.")
    return value


def _revision(value: str, label: str) -> str:
    if not value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError(f"{label} is empty or invalid.")
    return value


def _base_volume_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise ValueError(
            "--modal-base-volume-path must be an absolute, non-root POSIX path "
            "inside the mounted Volume and cannot contain '..'."
        )
    return str(path)


def gpu_count(value: str) -> int:
    match = GPU_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            "--modal-gpu must look like H100, B200:2, or another Modal GPU type "
            "with an optional count from 1 to 8."
        )
    return int(match.group(2) or "1")


def parse_server_args(
    values: list[str], reserved: frozenset[str] | set[str]
) -> dict[str, str]:
    """Parse repeated ``--flag`` / ``--flag=value`` serving-engine arguments.

    Runtime-agnostic: the caller supplies the set of flags that engine (or the
    wrapper) owns and therefore may not be overridden from the catalog.
    """
    parsed: dict[str, str] = {}
    for raw in values:
        if any(ord(character) < 32 for character in raw):
            raise ValueError("A serving-engine argument cannot contain control characters.")
        key, separator, value = raw.partition("=")
        if not SGLANG_KEY_PATTERN.fullmatch(key):
            raise ValueError(
                "Each --modal-sglang-arg must be --flag or --flag=value."
            )
        if key in reserved:
            raise ValueError(
                f"Serving argument {key!r} is owned by codex-modal and cannot be overridden."
            )
        if key in parsed:
            raise ValueError(f"Duplicate serving argument {key!r}.")
        parsed[key] = value if separator else ""
    return parsed


def parse_sglang_args(values: list[str]) -> dict[str, str]:
    """Back-compatible SGLang arg parser (uses SGLang's reserved flag set)."""
    return parse_server_args(values, RESERVED_SGLANG_ARGS)


def build_custom_deployment(
    options: WrapperOptions,
    resolved: ResolvedModel,
    app_name: str,
) -> CustomDeployment:
    if not options.self_managed or not options.gpu:
        raise ValueError("A self-managed deployment requires --modal-self-managed and --modal-gpu.")

    source_repo = _repo_id(options.custom_hf_repo or resolved.base_model, "Hugging Face repo")
    if options.custom_hf_repo:
        source_revision = _revision(
            options.custom_hf_revision or "main", "Custom Hugging Face revision"
        )
    else:
        source_revision = _revision(
            options.model_revision or "main", "Hugging Face model revision"
        )

    base_volume = options.base_volume
    base_path = _base_volume_path(options.base_volume_path) if options.base_volume_path else None
    if base_volume and not VOLUME_PATTERN.fullmatch(base_volume):
        raise ValueError("--modal-base-volume contains unsupported characters.")
    base_revision = (
        _revision(options.model_revision, "Base Hugging Face revision")
        if base_volume and options.model_revision
        else None
    )
    hf_token_env = options.hf_token_env
    if hf_token_env and not os.environ.get(hf_token_env):
        raise RuntimeError(f"Environment variable {hf_token_env!r} is empty or missing.")

    runtime = runtimes.runtime_for(options.runtime)
    image = options.sglang_image or runtime.default_image()
    if not image or any(character.isspace() or ord(character) < 32 for character in image):
        raise ValueError("--modal-sglang-image is empty or invalid.")
    server_args = parse_server_args(options.sglang_args, runtime.reserved_server_args())
    # Surface a bad download_subdir early (llama.cpp requires one).
    runtime.download_spec({"download_subdir": options.download_subdir})

    return CustomDeployment(
        app_name=app_name,
        volume_name=custom_volume_name(app_name),
        served_model=source_repo,
        source_repo=source_repo,
        source_revision=source_revision,
        base_repo=resolved.base_model if base_volume else None,
        base_revision=base_revision,
        base_volume=base_volume,
        base_volume_path=base_path,
        gpu=options.gpu,
        tensor_parallel_size=gpu_count(options.gpu),
        runtime=runtime.name,
        sglang_image=image,
        autoinference_utils_version=DEFAULT_AUTOINFERENCE_UTILS,
        server_args=server_args,
        serving_pip=tuple(options.serving_pip),
        download_subdir=options.download_subdir or None,
        cpu=options.cpu or DEFAULT_CPU,
        memory=options.memory or DEFAULT_MEMORY,
        scaledown_window=(
            DEFAULT_SCALEDOWN_WINDOW
            if options.scaledown_window is None
            else options.scaledown_window
        ),
        target_inputs=options.target_inputs or DEFAULT_TARGET_INPUTS,
        max_containers=DEFAULT_MAX_CONTAINERS,
        startup_timeout=options.startup_timeout,
        routing_region=options.routing_region,
        compute_regions=tuple(options.compute_regions),
        colocate_compute=options.colocate_compute,
        hf_token_env=hf_token_env,
    )
