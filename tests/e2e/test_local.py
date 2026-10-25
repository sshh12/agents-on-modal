"""End-to-end: the local sandbox (harness on the host) against a real Modal-served model.

Exercises the full path: gateway protocol translation, the harness config, the in-sandbox runner,
trace streaming, and a real tool-using task that must produce a verifiable artifact.
"""

from __future__ import annotations

import os

import pytest

from agents_on_modal import sandbox

HARNESSES = [h for h in os.environ.get("AOM_E2E_HARNESSES", "codex,claude").split(",") if h]


@pytest.mark.parametrize("harness", HARNESSES)
def test_local_file_task(endpoint, harness, tmp_path):
    """The agent must create a file with exact contents; we verify it on disk."""

    box = sandbox.create("local", workdir=str(tmp_path))
    try:
        run = box.start(
            harness,
            endpoint,
            "Create a file named DONE.txt in the current directory whose only contents are the "
            "word SUCCESS (no newline, nothing else). Then stop.",
            timeout=600,
        )
        kinds = []
        for event in run.events(follow=True, poll=2.0):
            kinds.append(event.kind)
        status = run.wait(timeout=60)

        assert status["state"] == "done", status
        artifact = tmp_path / "DONE.txt"
        assert artifact.is_file(), f"agent did not create DONE.txt; events seen: {set(kinds)}"
        assert artifact.read_text(encoding="utf-8").strip() == "SUCCESS"
        # the trace must carry real work, not just lifecycle
        assert {"tool_call", "text"} & set(kinds), kinds
    finally:
        box.stop()
