"""Shared fixtures for the opt-in end-to-end tests.

These spend real Modal credits and need a deployed model, the harness CLIs on PATH (for the local
sandbox), Docker (for the docker sandbox) and a Modal account. They are skipped unless ``AOM_E2E=1``.

    AOM_E2E=1 AOM_E2E_MODEL=qwen3.8-27b python -m pytest tests/e2e -v -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

#: This conftest lives in tests/e2e, but its collection hook receives the WHOLE session's items.
#: Only the e2e tests (those under this directory) are the opt-in, credit-spending ones -- scope
#: the skip to them, or a bare ``pytest`` run would skip every offline test too (and silently
#: "pass" while exercising nothing).
_E2E_DIR = pathlib.Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    if os.environ.get("AOM_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="e2e tests need AOM_E2E=1 (they spend Modal credits)")
    for item in items:
        try:
            item_path = pathlib.Path(str(getattr(item, "path", None) or item.fspath)).resolve()
        except Exception:  # noqa: BLE001 - an item without a filesystem path is never an e2e test
            continue
        if item_path == _E2E_DIR or _E2E_DIR in item_path.parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def model_name() -> str:
    return os.environ.get("AOM_E2E_MODEL", "qwen3.8-27b")


@pytest.fixture(scope="session")
def endpoint(model_name):
    """Attach to (or deploy) the model once and wait until its gateway answers."""

    from agents_on_modal import models

    model = models.get(model_name)
    ep = models.deploy(model) if os.environ.get("AOM_E2E_DEPLOY") == "1" else models.endpoint(model)
    models.wait_ready(ep, timeout=1800)
    return ep
