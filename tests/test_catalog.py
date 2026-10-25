"""The packaged catalog loads, round-trips, and the ported abliterated/uncensored family is intact."""

from __future__ import annotations

from agents_on_modal.models import Model, get, list_models

# Reserved engine flags the engine wrappers inject themselves; a catalog entry must not repeat them.
RESERVED = {
    "--host", "--port", "--model-path", "--served-model-name", "--tp", "--context-length",
    "--model", "-m", "--alias", "-a", "--n-gpu-layers", "-ngl", "--ctx-size", "-c",
}

PORTED = [
    "deepseek-v4-flash-abliterated",
    "glm-5.3-abliterated-nvfp4",
    "glm-5.3-uncensored-nvfp4",
    "deepseek-v4-flash-vision-uncensored",
    "deepseek-v4.1-flash-uncensored-fp8",
    "deepseek-v4.1-flash-abliterated-gguf",
]


def test_every_catalog_model_loads_and_round_trips():
    models = list_models()
    assert len(models) >= 11
    for m in models:
        assert Model.from_json(m.to_json()).name == m.name  # validates + round-trips


def test_ported_family_present():
    names = {m.name for m in list_models()}
    for name in PORTED:
        assert name in names, name


def test_no_reserved_engine_args_in_catalog():
    for m in list_models():
        for arg in m.engine_args:
            assert arg.split("=", 1)[0] not in RESERVED, (m.name, arg)


def test_base_volume_reuse_round_trips():
    m = get("deepseek-v4-flash-abliterated")
    assert m.base_repo == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert m.base_volume and m.base_volume_path == "/model"
    assert m.base_revision and m.revision  # abliterated checkpoint pinned to its own revision
    rt = Model.from_json(m.to_json())
    assert (rt.base_repo, rt.base_volume, rt.base_revision, rt.base_volume_path) == (
        m.base_repo, m.base_volume, m.base_revision, m.base_volume_path
    )


def test_vision_flag_only_on_vision_model():
    assert get("deepseek-v4-flash-vision-uncensored").vision is True
    assert get("deepseek-v4-flash-abliterated").vision is False


def test_gguf_entry_uses_llamacpp_with_subdir_and_no_image():
    m = get("deepseek-v4.1-flash-abliterated-gguf")
    assert m.engine == "llamacpp"
    assert m.download_subdir == "Mix-256G"
    assert m.image is None  # the llamacpp engine builds its own CUDA base
    assert m.max_output_tokens < m.context_window
