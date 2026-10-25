"""Shared fixtures for the opt-in end-to-end tests.

These spend real Modal credits and need a deployed model, the harness CLIs on PATH (for the local
sandbox), Docker (for the docker sandbox) and a Modal account. They are skipped unless ``AOM_E2E=1``.

    AOM_E2E=1 AOM_E2E_MODEL=qwen3.8-27b python -m pytest tests/e2e -v -s
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("AOM_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="e2e tests need AOM_E2E=1 (they spend Modal credits)")
    for item in items:
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
