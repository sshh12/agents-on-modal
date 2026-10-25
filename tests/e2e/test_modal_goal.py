"""End-to-end: goal mode in the modal sandbox for both harnesses.

Goal mode must (a) pursue a multi-step goal across turns and (b) *terminate* once the goal is met --
not loop until the deadline. Claude drives its native in-process ``/goal`` loop; Codex is driven by
the runner, which stops resuming when the model signals completion (a ``GOAL_COMPLETE`` sentinel
line) or stops taking new distinct actions. Skipped unless ``AOM_E2E=1``.
"""

from __future__ import annotations

import os

import pytest

from agents_on_modal import sandbox

HARNESSES = [h for h in os.environ.get("AOM_E2E_HARNESSES", "codex,claude").split(",") if h]

GOAL_TASK = (
    "Goal: create a file /work/steps.txt that contains exactly three lines, in order: one, two, "
    "three. Build it up one line at a time (append a line, then read the file to check), and only "
    "finish once the file has all three lines in the right order and you have verified it."
)


@pytest.mark.parametrize("harness", HARNESSES)
def test_modal_goal(endpoint, harness):
    box = sandbox.create("modal", endpoint=endpoint)
    try:
        run = box.start(harness, endpoint, GOAL_TASK, goal=True, timeout=600)
        for _ in run.events(follow=True, poll=3.0):
            pass
        status = run.wait(timeout=60)

        # The goal must terminate on its own (done), not be killed by the deadline (timeout).
        assert status["state"] == "done", status
        content = box.read_file("/work/steps.txt").decode("utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        assert lines[:3] == ["one", "two", "three"], content
    finally:
        box.stop()
