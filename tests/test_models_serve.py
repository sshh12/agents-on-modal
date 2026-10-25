"""Unit tests for serve.py: offline-importable, config reader round-trips, port wiring."""

from __future__ import annotations

import json

import agents_on_modal.models.lifecycle as deploy
import agents_on_modal.models.serve as serve
from agents_on_modal.models import Model, get


def test_serve_imports_offline_with_stub():
    # Imported with no AOM_MODEL_JSON in the environment -> the stub drives the graph.
    assert serve.engine.name == "sglang"
    assert serve.MODEL_PATH == "/aom-weights/model"
    assert serve.ENGINE_PORT == 8001
    assert serve.SERVER_PORT == 8000


def test_serve_env_var_names_match_deploy():
    # deploy.py duplicates these constants; drift would silently break deploys.
    assert serve.AOM_MODEL_JSON_ENV == deploy.AOM_MODEL_JSON_ENV
    assert serve.AOM_APP_NAME_ENV == deploy.AOM_APP_NAME_ENV


def test_load_model_json_reads_env(monkeypatch):
    model = get("qwen3.8-27b")
    payload = model.to_json()
    monkeypatch.setenv(serve.AOM_MODEL_JSON_ENV, json.dumps(payload))
    loaded = serve._load_model_json()
    assert loaded == payload
    # and it round-trips back into a Model
    assert Model.from_json(loaded).name == model.name


def test_load_model_json_falls_back_to_stub(monkeypatch):
    monkeypatch.delenv(serve.AOM_MODEL_JSON_ENV, raising=False)
    loaded = serve._load_model_json()
    assert loaded["engine"] == "sglang"
    assert loaded["name"] == "aom-import-stub"


def test_volume_name_derived_from_model_name():
    # module-level volume name must mirror Model.volume_name for the stub
    assert serve.volume_name == "aom-aom-import-stub-weights"


def test_model_volume_name_contract_for_real_model():
    model = get("qwen3.8-27b")
    assert model.volume_name == "aom-qwen3-8-27b-weights"
    # serve derives the same shape from the model name (independent of app shortening)
    derived = f"aom-{model.name.replace('.', '-')}-weights"
    assert derived == model.volume_name
