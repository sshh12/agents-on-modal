"""End-to-end: the modal sandbox (harness INSIDE a Modal container) against a real model.

Exercises the full remote path: the batteries image, Sandbox.create with a run-scoped proxy token,
writing the harness config through the filesystem API, the in-sandbox runner launched via ``gosu
agent``, trace streaming back through the filesystem API, and a real tool-using task whose artifact
is verified by reading it back out of the container.

Assumes the batteries image is published (``ModalSandbox.ensure_image(build=True)`` once) and a model
is deployed; both are true after a normal ``aom`` setup. Skipped unless ``AOM_E2E=1``.
"""

from __future__ import annotations

import os

import pytest

from agents_on_modal import sandbox

HARNESSES = [h for h in os.environ.get("AOM_E2E_HARNESSES", "codex,claude").split(",") if h]


@pytest.mark.parametrize("harness", HARNESSES)
def test_modal_file_task(endpoint, harness):
    """The agent runs inside a Modal sandbox and must create a verifiable artifact on /work."""

    box = sandbox.create("modal", endpoint=endpoint)
    try:
        run = box.start(
            harness,
            endpoint,
            "Create a file named DONE.txt in the current directory whose only contents are the "
            "word SUCCESS (no newline, nothing else). Then stop.",
            timeout=600,
        )
        kinds = []
        for event in run.events(follow=True, poll=3.0):
            kinds.append(event.kind)
        status = run.wait(timeout=60)

        assert status["state"] == "done", status
        artifact = box.read_file("/work/DONE.txt").decode("utf-8")
        assert artifact.strip() == "SUCCESS", artifact
        # the trace must carry real work, not just lifecycle
        assert {"tool_call", "text"} & set(kinds), kinds
    finally:
        box.stop()
