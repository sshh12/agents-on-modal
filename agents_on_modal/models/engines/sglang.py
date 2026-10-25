"""SGLang engine: a prebuilt registry image, ``python -m sglang.launch_server``.

The wrapper owns the bind address, port, model path, served name, tensor-parallel
size and context length; everything else comes from the catalog's ``engine_args``.
SGLang reads the checkpoint's ``config.json`` to build the model, so the whole repo
is downloaded. ``aiohttp`` is layered onto the image because the gateway runs in the
same container and the SGLang base image does not ship it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Engine, DownloadSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import modal

#: Base image used when a catalog entry does not name one.
DEFAULT_IMAGE = "lmsysorg/sglang:v0.5.22-cu130"


class SGLangEngine(Engine):
    name = "sglang"

    def image(self, model_json: dict[str, Any]) -> "modal.Image":
        import modal

        image_ref = str(model_json.get("image") or DEFAULT_IMAGE)
        image = modal.Image.from_registry(image_ref).uv_pip_install("aiohttp>=3.9,<4")
        # Extra wheels last, so an explicit pin wins over the base image (e.g. a
        # transformers new enough to recognise a very new model architecture).
        extra = tuple(str(p) for p in (model_json.get("serving_pip") or ()))
        if extra:
            image = image.uv_pip_install(*extra)
        return image

    def download_spec(self, model_json: dict[str, Any]) -> DownloadSpec:
        # Whole repo (or a subdir); SGLang needs config.json plus every shard.
        return DownloadSpec(subdir=model_json.get("download_subdir"), needs_config_json=True)

    def reserved_args(self) -> frozenset[str]:
        return frozenset(
            {"--host", "--model-path", "--port", "--served-model-name", "--tp", "--context-length"}
        )

    def launch_argv(self, model_json: dict[str, Any], model_path: str, port: int) -> list[str]:
        from .base import gpu_count

        argv = [
            "python",
            "-m",
            "sglang.launch_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(model_path),
            "--served-model-name",
            self.served_model_name(model_json),
            "--tp",
            str(gpu_count(str(model_json["gpu"]))),
            "--context-length",
            str(int(model_json["context_window"])),
        ]
        argv += self._checked_engine_args(model_json)
        return argv

    def ready_url(self, port: int) -> str:
        # SGLang's /health runs a 1-token generation, so 200 means it can serve.
        return f"http://127.0.0.1:{port}/health"
