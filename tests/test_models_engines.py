"""Unit tests for the engine strategies (no Modal, no GPU, no network)."""

from __future__ import annotations

import pytest

import agents_on_modal.models.engines as engines
from agents_on_modal.config import AomError
from agents_on_modal.models import Model


def _sglang_json() -> dict:
    return {
        "name": "qwen3.8-27b",
        "repo": "Qwen/Qwen3.8-27B-FP8",
        "engine": "sglang",
        "gpu": "H100",
        "context_window": 262144,
        "max_output_tokens": 32768,
        "image": "lmsysorg/sglang:v0.5.22-cu130",
        "engine_args": ["--reasoning-parser=qwen3", "--tool-call-parser=qwen3_coder"],
        "engine_env": {"SGLANG_DEFAULT_THINKING": "1"},
        "serving_pip": ["transformers==5.16.1"],
    }


def _llamacpp_json(**overrides) -> dict:
    data = {
        "name": "deepseek-v41-gguf",
        "repo": "unsloth/DeepSeek-V4.1-Flash-GGUF",
        "engine": "llamacpp",
        "gpu": "H200:2",
        "context_window": 131072,
        "max_output_tokens": 16384,
        "download_subdir": "Q4_K_M",
        "engine_args": ["--parallel=4"],
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# registry                                                                     #
# --------------------------------------------------------------------------- #
def test_registry_get_and_names():
    assert engines.get("sglang").name == "sglang"
    assert engines.get("SGLang").name == "sglang"  # case-insensitive
    assert engines.get("llamacpp").name == "llamacpp"
    assert engines.names() == ("llamacpp", "sglang")


def test_registry_unknown_raises():
    with pytest.raises(AomError):
        engines.get("vllm")


# --------------------------------------------------------------------------- #
# SGLang                                                                       #
# --------------------------------------------------------------------------- #
def test_sglang_launch_argv():
    engine = engines.get("sglang")
    argv = engine.launch_argv(_sglang_json(), "/aom-weights/model", 8001)
    assert argv[:3] == ["python", "-m", "sglang.launch_server"]

    def flag(name):
        return argv[argv.index(name) + 1]

    assert flag("--host") == "127.0.0.1"
    assert flag("--port") == "8001"
    assert flag("--model-path") == "/aom-weights/model"
    assert flag("--served-model-name") == "qwen3.8-27b"
    assert flag("--tp") == "1"
    assert flag("--context-length") == "262144"
    # catalog engine_args appended verbatim
    assert "--reasoning-parser=qwen3" in argv
    assert "--tool-call-parser=qwen3_coder" in argv


def test_sglang_tp_from_gpu_count():
    model_json = dict(_sglang_json(), gpu="H200:2")
    argv = engines.get("sglang").launch_argv(model_json, "/m", 8001)
    assert argv[argv.index("--tp") + 1] == "2"


def test_sglang_reserved_arg_collision_raises():
    model_json = dict(_sglang_json(), engine_args=["--tp=4"])
    with pytest.raises(AomError):
        engines.get("sglang").launch_argv(model_json, "/m", 8001)


def test_sglang_env_merges_base_and_engine_env():
    env = engines.get("sglang").env(_sglang_json())
    assert env["DO_NOT_TRACK"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert env["SGLANG_DEFAULT_THINKING"] == "1"  # per-model override present


def test_sglang_download_spec_whole_repo_with_config():
    spec = engines.get("sglang").download_spec(_sglang_json())
    assert spec.subdir is None
    assert spec.needs_config_json is True


def test_sglang_reserved_args():
    assert engines.get("sglang").reserved_args() == frozenset(
        {"--host", "--model-path", "--port", "--served-model-name", "--tp", "--context-length"}
    )


def test_sglang_ready_url():
    assert engines.get("sglang").ready_url(8001) == "http://127.0.0.1:8001/health"


# --------------------------------------------------------------------------- #
# llama.cpp                                                                    #
# --------------------------------------------------------------------------- #
def test_llamacpp_launch_argv_picks_first_shard(tmp_path):
    model_root = tmp_path / "model"
    quant = model_root / "Q4_K_M"
    quant.mkdir(parents=True)
    # Out-of-order shards + an imatrix file that must be ignored.
    (quant / "m-00002-of-00002.gguf").write_bytes(b"x")
    (quant / "m-00001-of-00002.gguf").write_bytes(b"x")
    (quant / "imatrix.gguf").write_bytes(b"x")

    argv = engines.get("llamacpp").launch_argv(_llamacpp_json(), str(model_root), 8001)

    def flag(name):
        return argv[argv.index(name) + 1]

    assert argv[0].endswith("llama-server")
    assert flag("--model").endswith("m-00001-of-00002.gguf")
    assert flag("--alias") == "deepseek-v41-gguf"
    assert flag("--host") == "127.0.0.1"
    assert flag("--port") == "8001"
    assert flag("--n-gpu-layers") == "99"
    assert flag("--ctx-size") == "131072"
    assert "--parallel=4" in argv


def test_llamacpp_launch_argv_lone_gguf(tmp_path):
    quant = tmp_path / "model" / "Q4_K_M"
    quant.mkdir(parents=True)
    (quant / "solo.gguf").write_bytes(b"x")
    argv = engines.get("llamacpp").launch_argv(_llamacpp_json(), str(tmp_path / "model"), 8001)
    assert argv[argv.index("--model") + 1].endswith("solo.gguf")


def test_llamacpp_launch_argv_no_gguf_raises(tmp_path):
    (tmp_path / "model" / "Q4_K_M").mkdir(parents=True)
    with pytest.raises(AomError):
        engines.get("llamacpp").launch_argv(_llamacpp_json(), str(tmp_path / "model"), 8001)


def test_llamacpp_download_spec_requires_subdir():
    engine = engines.get("llamacpp")
    spec = engine.download_spec(_llamacpp_json())
    assert spec.subdir == "Q4_K_M"
    assert spec.needs_config_json is False
    with pytest.raises(AomError):
        engine.download_spec(_llamacpp_json(download_subdir=None))


def test_llamacpp_reserved_args_include_context_and_model():
    reserved = engines.get("llamacpp").reserved_args()
    for flag in ("--host", "--port", "--model", "-m", "--alias", "-a",
                 "--n-gpu-layers", "-ngl", "--ctx-size", "-c"):
        assert flag in reserved


def test_llamacpp_env_and_ready_url():
    engine = engines.get("llamacpp")
    assert engine.ready_url(8001) == "http://127.0.0.1:8001/health"
    assert engine.env(_llamacpp_json())["PYTHONUNBUFFERED"] == "1"


# --------------------------------------------------------------------------- #
# round trip through the real Model dataclass                                  #
# --------------------------------------------------------------------------- #
def test_launch_argv_from_catalog_model():
    model = Model.from_json(
        {
            "name": "qwen3.8-27b",
            "description": "d",
            "repo": "Qwen/Qwen3.8-27B-FP8",
            "engine": "sglang",
            "gpu": "H100",
            "context_window": 262144,
            "max_output_tokens": 32768,
            "reasoning": {"default": "medium"},
        }
    )
    argv = engines.get(model.engine).launch_argv(model.to_json(), "/m", 8001)
    assert "--served-model-name" in argv and argv[argv.index("--served-model-name") + 1] == "qwen3.8-27b"


def test_engine_image_builds_lazily():
    modal = pytest.importorskip("modal")
    image = engines.get("sglang").image(_sglang_json())
    assert isinstance(image, modal.Image)
