"""End-to-end: the docker sandbox (agent in a firewalled local container, model via egress broker).

Exercises the two-container topology: the agent runs on an ``--internal`` network with a default-deny
firewall and no usable credential; its model traffic reaches the gateway only through the egress
broker (which injects the real token and stays direct), and its code/browser traffic only through the
broker's proxy. Needs a running Docker daemon and the batteries image built
(``ensure_image(build=True)`` once). Skipped unless ``AOM_E2E=1``.
"""

from __future__ import annotations

import os

import pytest

from agents_on_modal import sandbox

HARNESSES = [h for h in os.environ.get("AOM_E2E_DOCKER_HARNESSES", "codex,claude").split(",") if h]


@pytest.mark.parametrize("harness", HARNESSES)
def test_docker_file_task(endpoint, harness):
    box = sandbox.create("docker", endpoint=endpoint)
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
        assert {"tool_call", "text"} & set(kinds), kinds
    finally:
        box.stop()
