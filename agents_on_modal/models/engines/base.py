"""The ``Engine`` strategy: the three things SGLang and llama.cpp differ in.

An engine turns a model's JSON spec (``Model.to_json()``) into:

  1. a container **image** (a prebuilt registry image vs. a CUDA base that compiles
     llama.cpp from source),
  2. what to **download** from the repo and how completeness is proven
     (HF-native safetensors with a root ``config.json`` vs. a subdirectory of GGUF
     shards with no config), and
  3. how the server subprocess is **launched** and health-checked.

Only ``image()`` touches Modal, and it imports ``modal`` lazily, so this module and
its subclasses import cleanly off-host with nothing but the standard library.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...config import AomError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import modal

#: GPU spec ``TYPE`` or ``TYPE:N`` (tensor-parallel size = N); mirrors models.GPU_PATTERN.
GPU_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_-]{0,31})(?::([1-8]))?$")

#: Non-secret runtime environment shared by every engine subprocess.
BASE_RUNTIME_ENV: dict[str, str] = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


@dataclass(frozen=True)
class DownloadSpec:
    """What the CPU ``prepare`` step must fetch and how it proves completeness."""

    #: Restrict the repo download to this subdirectory (``None`` = whole repo).
    subdir: str | None
    #: True for HF-native safetensors (SGLang reads ``config.json``); False for GGUF
    #: (self-describing, no ``config.json`` - completeness is "at least one .gguf").
    needs_config_json: bool


def gpu_count(gpu: str) -> int:
    """Tensor-parallel size implied by a GPU spec (``H100`` -> 1, ``B200:2`` -> 2)."""

    match = GPU_PATTERN.fullmatch(str(gpu or ""))
    return int(match.group(2) or "1") if match else 1


def arg_key(raw: str) -> str:
    """The flag portion of an ``--flag`` / ``--flag=value`` engine argument."""

    return raw.split("=", 1)[0]


class Engine(ABC):
    """Per-engine serving strategy. Subclasses are stateless and cheap to construct."""

    #: Registry key (``sglang`` / ``llamacpp``); matches ``Model.engine``.
    name: str = ""

    # ---- deploy-time (runs on the host / at image build) --------------------
    @abstractmethod
    def image(self, model_json: dict[str, Any]) -> "modal.Image":
        """Build the serving container image for this model."""

    @abstractmethod
    def download_spec(self, model_json: dict[str, Any]) -> DownloadSpec:
        """What ``prepare`` downloads and how it verifies completeness."""

    def reserved_args(self) -> frozenset[str]:
        """Engine flags the wrapper owns; a catalog ``engine_args`` may not set them."""

        return frozenset()

    # ---- serve-time (runs inside the GPU container) -------------------------
    @abstractmethod
    def launch_argv(self, model_json: dict[str, Any], model_path: str, port: int) -> list[str]:
        """The argv that starts the engine bound to ``127.0.0.1:<port>``."""

    def env(self, model_json: dict[str, Any]) -> dict[str, str]:
        """Environment for the engine subprocess: shared defaults + per-model overrides."""

        engine_env = {str(k): str(v) for k, v in (model_json.get("engine_env") or {}).items()}
        return {**BASE_RUNTIME_ENV, **engine_env}

    @abstractmethod
    def ready_url(self, port: int) -> str:
        """A URL that returns 200 only once the engine can actually generate."""

    # ---- shared helpers ------------------------------------------------------
    def _checked_engine_args(self, model_json: dict[str, Any]) -> list[str]:
        """Return the catalog ``engine_args`` after rejecting reserved-flag collisions."""

        reserved = self.reserved_args()
        args = [str(a) for a in (model_json.get("engine_args") or ())]
        for raw in args:
            if arg_key(raw) in reserved:
                raise AomError(
                    f"engine_args entry {raw!r} sets {arg_key(raw)!r}, which the "
                    f"{self.name} engine owns and sets itself."
                )
        return args

    def served_model_name(self, model_json: dict[str, Any]) -> str:
        """The id the engine advertises on ``/v1/models`` (the catalog name)."""

        return str(model_json["name"])
