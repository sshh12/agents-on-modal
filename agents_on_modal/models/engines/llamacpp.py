"""llama.cpp engine: a CUDA devel base that compiles ``llama-server`` from a pinned PR.

DeepSeek-V4.1's architecture is not in a tagged llama.cpp release yet, so the GGUFs
require the exact PR/commit the model card names; stock llama.cpp, Ollama and LM Studio
refuse to load them. The image, the driver-stub linker trick and the GGUF shard picker
are ported verbatim from the old ``codex_modal/runtimes.py`` (commit ``c4872ed``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...config import AomError
from .base import Engine, DownloadSpec, gpu_count

if TYPE_CHECKING:  # pragma: no cover - typing only
    import modal

# llama.cpp source build (pinned to the open PR that adds DeepSeek-V4.1-Flash).
LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMACPP_PR = "30817"
LLAMACPP_COMMIT = "b72050d03"
DEFAULT_CUDA_BASE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
LLAMACPP_BIN = "/opt/llama.cpp/build/bin/llama-server"

# CMAKE_CUDA_ARCHITECTURES cannot be auto-detected at image-build time (no GPU is
# attached), so it is derived from the deploy's GPU type.
_CUDA_ARCH = {
    "t4": "75", "l4": "89", "l40s": "89", "a10": "86", "a10g": "86",
    "a100": "80", "a100-40gb": "80", "a100-80gb": "80",
    "h100": "90", "h200": "90",
    "b200": "100", "b300": "100", "gb200": "100", "gb300": "100",
}
_DEFAULT_CUDA_ARCH = "90"


def _first_gguf(directory: Path) -> Path:
    """Pick the entry shard llama.cpp should open (it auto-loads the rest)."""

    first_shard = sorted(directory.glob("*-00001-of-*.gguf"))
    if first_shard:
        return first_shard[0]
    lone = sorted(p for p in directory.glob("*.gguf") if "imatrix" not in p.name.lower())
    if not lone:
        raise AomError(f"No .gguf files found under {directory}.")
    return lone[0]


class LlamaCppEngine(Engine):
    name = "llamacpp"

    def image(self, model_json: dict[str, Any]) -> "modal.Image":
        import modal

        image_ref = str(model_json.get("image") or DEFAULT_CUDA_BASE)
        arch = _CUDA_ARCH.get(str(model_json["gpu"]).split(":", 1)[0].lower(), _DEFAULT_CUDA_ARCH)
        image = (
            modal.Image.from_registry(image_ref, add_python="3.12")
            .apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev", "ccache")
            .run_commands(
                f"git clone {LLAMACPP_REPO} /opt/llama.cpp",
                f"cd /opt/llama.cpp && git fetch origin pull/{LLAMACPP_PR}/head:pr "
                f"&& git checkout {LLAMACPP_COMMIT}",
                # The CUDA -devel image ships the toolkit but not libcuda (the real
                # driver is injected by Modal only at run time), so a naive build
                # fails to link against the driver API. Point the linker at the
                # driver STUB: CMAKE_LIBRARY_PATH lets CUDA::cuda_driver resolve, and
                # the symlink satisfies the libcuda.so.1 SONAME recorded in
                # libggml-cuda.so. The stub is build-time only; the real driver wins
                # at run time.
                "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "
                "/usr/local/cuda/lib64/stubs/libcuda.so.1",
                "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON "
                f"-DCMAKE_CUDA_ARCHITECTURES={arch} "
                "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
                "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs "
                "&& cmake --build build -j --config Release --target llama-server",
            )
            .uv_pip_install("aiohttp>=3.9,<4")
        )
        extra = tuple(str(p) for p in (model_json.get("serving_pip") or ()))
        if extra:
            image = image.uv_pip_install(*extra)
        return image

    def download_spec(self, model_json: dict[str, Any]) -> DownloadSpec:
        subdir = model_json.get("download_subdir")
        if not subdir:
            raise AomError(
                "llama.cpp models require 'download_subdir' naming the GGUF quant folder."
            )
        return DownloadSpec(subdir=str(subdir), needs_config_json=False)

    def reserved_args(self) -> frozenset[str]:
        # The wrapper owns the model path, alias, bind address, GPU offload and context.
        return frozenset(
            {"--host", "--port", "--model", "-m", "--alias", "-a",
             "--n-gpu-layers", "-ngl", "--ctx-size", "-c"}
        )

    def launch_argv(self, model_json: dict[str, Any], model_path: str, port: int) -> list[str]:
        subdir = model_json.get("download_subdir")
        if not subdir:
            raise AomError("llama.cpp models require 'download_subdir'.")
        gguf_dir = Path(model_path) / str(subdir)
        entry = _first_gguf(gguf_dir)
        argv = [
            LLAMACPP_BIN,
            "--model",
            str(entry),
            "--alias",
            self.served_model_name(model_json),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--n-gpu-layers",
            "99",
            "--ctx-size",
            str(int(model_json["context_window"])),
        ]
        argv += self._checked_engine_args(model_json)
        return argv

    def ready_url(self, port: int) -> str:
        # llama-server /health is 200 {"status":"ok"} only once the model is loaded.
        return f"http://127.0.0.1:{port}/health"
