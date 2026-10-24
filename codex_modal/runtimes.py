"""Serving-runtime strategies for the generic self-managed Modal app.

The wrapper can serve a checkpoint with more than one inference engine. The
engines differ in only three places:

  1. the container image (a prebuilt SGLang registry image vs. a CUDA base that
     compiles llama.cpp from source),
  2. what has to be downloaded from the repo and how "ready to serve" is proven
     (native safetensors with a root ``config.json`` vs. a subdirectory of GGUF
     shards with no config at all), and
  3. how the server process is launched and health-checked.

Bundling those three per-engine decisions behind one small ``Runtime`` interface
keeps ``custom_modal_app`` a thin Modal-wiring layer and keeps ``custom_deploy``
free of engine-specific branching: adding an engine is a new ``Runtime``
subclass here, not an ``if`` threaded through both modules.

Only ``modal`` is imported at module load (image construction happens locally at
deploy time). Engine SDKs and the server launchers are imported lazily inside
the container-only ``start`` methods so this module stays importable anywhere.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import modal


SGLANG = "sglang"
LLAMACPP = "llamacpp"
DEFAULT_RUNTIME = SGLANG
KNOWN_RUNTIMES = (SGLANG, LLAMACPP)

# llama.cpp source build. DeepSeek-V4.1's architecture is not in a tagged
# release yet, so these GGUFs require the exact PR/commit the model card names;
# stock llama.cpp, Ollama and LM Studio refuse to load them.
LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMACPP_PR = "30817"
LLAMACPP_COMMIT = "b72050d03"
LLAMACPP_CUDA_BASE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
LLAMACPP_BIN = "/opt/llama.cpp/build/bin/llama-server"

# CMAKE_CUDA_ARCHITECTURES cannot be auto-detected at image-build time (no GPU
# is attached), so it is derived from the deploy's GPU type.
_CUDA_ARCH = {
    "t4": "75", "l4": "89", "l40s": "89", "a10": "86", "a10g": "86",
    "a100": "80", "a100-40gb": "80", "a100-80gb": "80",
    "h100": "90", "h200": "90",
    "b200": "100", "b300": "100", "gb200": "100", "gb300": "100",
}
_DEFAULT_CUDA_ARCH = "90"


@dataclass(frozen=True)
class DownloadSpec:
    """What ``prepare_model`` must fetch and how it proves completeness."""

    # Restrict the repo download to this subdirectory (``None`` = whole repo).
    subdir: str | None
    # Whether a root ``config.json`` must exist for the download to count as
    # complete. True for HF-native safetensors (SGLang reads config.json to
    # build the model); False for GGUF, which is self-describing and ships no
    # config.json - there, completeness is "at least one .gguf landed".
    needs_config_json: bool


def _first_gguf(directory: Any) -> Any:
    """Pick the entry shard llama.cpp should open (it auto-loads the rest).

    ``directory`` is a ``pathlib.Path``; the caller imports Path in-container.
    """
    first_shard = sorted(directory.glob("*-00001-of-*.gguf"))
    if first_shard:
        return first_shard[0]
    lone = sorted(p for p in directory.glob("*.gguf") if "imatrix" not in p.name.lower())
    if not lone:
        raise RuntimeError(f"No .gguf files found under {directory}.")
    return lone[0]


def _http_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    method = "POST" if payload is not None else "GET"
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


class Runtime:
    """Per-engine serving strategy. Subclasses are stateless and cheap."""

    name: str = ""

    # ---- deploy-time (local) -------------------------------------------------
    def default_image(self) -> str:
        """Base registry image used when the catalog does not name one."""
        raise NotImplementedError

    def build_serving_image(self, image_ref: str, config: dict[str, Any]) -> "modal.Image":
        """Construct the serving image from its base registry ref.

        The runtime owns ``from_registry`` because the options differ: an SGLang
        image already ships Python, while a bare CUDA base needs ``add_python``.
        """
        raise NotImplementedError

    def reserved_server_args(self) -> frozenset[str]:
        """Server flags the wrapper sets itself; the catalog may not override."""
        return frozenset()

    def download_spec(self, config: dict[str, Any]) -> DownloadSpec:
        raise NotImplementedError

    # ---- serve-time (inside the container) -----------------------------------
    def start(self, config: dict[str, Any], model_path: str, port: int) -> "ServerHandle":
        raise NotImplementedError

    def ready_message(self, config: dict[str, Any]) -> str:
        return f"{config['served_model']} ({config['gpu']}) {self.name} deployment is ready."


class ServerHandle:
    """Opaque handle returned by ``Runtime.start`` so the app can stop it."""

    def stop(self) -> None:  # pragma: no cover - trivial
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# SGLang                                                                        #
# --------------------------------------------------------------------------- #
class _SGLangHandle(ServerHandle):
    def __init__(self, endpoint: Any) -> None:
        self._endpoint = endpoint

    def stop(self) -> None:
        self._endpoint.stop()


class SGLangRuntime(Runtime):
    name = SGLANG

    def default_image(self) -> str:
        from .custom_deploy import DEFAULT_SGLANG_IMAGE

        return DEFAULT_SGLANG_IMAGE

    def build_serving_image(self, image_ref: str, config: dict[str, Any]) -> "modal.Image":
        return modal.Image.from_registry(image_ref).uv_pip_install(
            f"autoinference-utils=={config['autoinference_utils_version']}"
        )

    def reserved_server_args(self) -> frozenset[str]:
        return frozenset(
            {"--host", "--model-path", "--port", "--served-model-name", "--tp"}
        )

    def download_spec(self, config: dict[str, Any]) -> DownloadSpec:
        # Whole repo; SGLang needs config.json plus every shard.
        return DownloadSpec(subdir=config.get("download_subdir"), needs_config_json=True)

    def start(self, config: dict[str, Any], model_path: str, port: int) -> ServerHandle:
        from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions

        extra_server_args = {"--served-model-name": str(config["served_model"])} | {
            str(key): str(value) for key, value in config["server_args"].items()
        }
        endpoint = SGLangEndpoint(
            model_path=model_path,
            worker_port=port,
            tp=int(config["tensor_parallel_size"]),
            extra_server_args=extra_server_args,
            health_timeout=int(config["startup_timeout"]),
            health_poll_interval=5.0,
        )
        endpoint.start()
        warmup_chat_completions(
            port=port,
            payload={
                "model": str(config["served_model"]),
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 8,
                "temperature": 0,
            },
            successful_requests=2,
            request_timeout=60.0,
        )
        return _SGLangHandle(endpoint)


# --------------------------------------------------------------------------- #
# llama.cpp                                                                     #
# --------------------------------------------------------------------------- #
class _LlamaCppHandle(ServerHandle):
    def __init__(self, process: Any) -> None:
        self._process = process

    def stop(self) -> None:
        import contextlib

        self._process.terminate()
        with contextlib.suppress(Exception):
            self._process.wait(timeout=20)
        if self._process.poll() is None:
            self._process.kill()


class LlamaCppRuntime(Runtime):
    name = LLAMACPP

    def default_image(self) -> str:
        return LLAMACPP_CUDA_BASE

    def build_serving_image(self, image_ref: str, config: dict[str, Any]) -> "modal.Image":
        arch = _CUDA_ARCH.get(str(config["gpu"]).split(":", 1)[0].lower(), _DEFAULT_CUDA_ARCH)
        return (
            modal.Image.from_registry(image_ref, add_python="3.12")
            .apt_install(
                "git", "cmake", "build-essential", "libcurl4-openssl-dev", "ccache"
            )
            .run_commands(
                f"git clone {LLAMACPP_REPO} /opt/llama.cpp",
                f"cd /opt/llama.cpp && git fetch origin pull/{LLAMACPP_PR}/head:pr "
                f"&& git checkout {LLAMACPP_COMMIT}",
                # The CUDA -devel image ships the toolkit but not libcuda (the
                # real driver is injected by Modal only at run time), so a naive
                # build fails to link llama-server against the driver API
                # (undefined cuMem*/cuDevice*). Point the linker at the driver
                # STUB: CMAKE_LIBRARY_PATH lets CUDA::cuda_driver resolve, and
                # the symlink satisfies the libcuda.so.1 SONAME recorded in
                # libggml-cuda.so. The stub is build-time only; it is not on the
                # runtime loader path, so the real driver still wins at run time.
                "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "
                "/usr/local/cuda/lib64/stubs/libcuda.so.1",
                "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON "
                f"-DCMAKE_CUDA_ARCHITECTURES={arch} "
                "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
                "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs "
                "&& cmake --build build -j --config Release --target llama-server",
            )
        )

    def reserved_server_args(self) -> frozenset[str]:
        # The wrapper owns the model path, alias, bind address and GPU offload.
        return frozenset(
            {"--host", "--port", "--model", "-m", "--alias", "-a",
             "--n-gpu-layers", "-ngl"}
        )

    def download_spec(self, config: dict[str, Any]) -> DownloadSpec:
        subdir = config.get("download_subdir")
        if not subdir:
            raise RuntimeError(
                "llama.cpp runtime requires 'download_subdir' naming the GGUF quant "
                "folder to serve (e.g. 'Mix-256G')."
            )
        return DownloadSpec(subdir=subdir, needs_config_json=False)

    def start(self, config: dict[str, Any], model_path: str, port: int) -> ServerHandle:
        import subprocess
        from pathlib import Path

        gguf_dir = Path(model_path) / str(config["download_subdir"])
        entry = _first_gguf(gguf_dir)

        command = [
            LLAMACPP_BIN,
            "--model", str(entry),
            "--alias", str(config["served_model"]),
            "--host", "0.0.0.0",
            "--port", str(port),
            "--n-gpu-layers", "99",
        ]
        for key, value in config["server_args"].items():
            command.append(str(key))
            if value != "":
                command.append(str(value))

        print(f"[llama.cpp] launching: {' '.join(command)}", flush=True)
        process = subprocess.Popen(command)
        self._await_health(process, port, int(config["startup_timeout"]))
        self._warmup(config, port)
        return _LlamaCppHandle(process)

    @staticmethod
    def _await_health(process: Any, port: int, timeout: int) -> None:
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{port}/health"
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {process.returncode} before readiness."
                )
            try:
                _http_json(url, None, timeout=10.0)
                return
            except urllib.error.HTTPError as error:
                if error.code not in (503, 500):  # 503 = still loading
                    raise
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(5.0)
        raise RuntimeError(f"llama-server did not become healthy within {timeout}s.")

    @staticmethod
    def _warmup(config: dict[str, Any], port: int) -> None:
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        payload = {
            "model": str(config["served_model"]),
            "messages": [{"role": "user", "content": "Reply with exactly OK."}],
            "max_tokens": 8,
            "temperature": 0,
        }
        for _ in range(2):
            _http_json(url, payload, timeout=120.0)


_RUNTIMES: dict[str, Runtime] = {SGLANG: SGLangRuntime(), LLAMACPP: LlamaCppRuntime()}


def runtime_for(name: str | None) -> Runtime:
    key = (name or DEFAULT_RUNTIME).lower()
    try:
        return _RUNTIMES[key]
    except KeyError:
        raise ValueError(
            f"Unknown runtime {name!r}. Known runtimes: {', '.join(KNOWN_RUNTIMES)}."
        ) from None
