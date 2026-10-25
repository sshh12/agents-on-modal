"""End-to-end: the dynamic-swarm controller drives real Modal sandboxes through the shared volume.

This bypasses the LLM orchestrator (which is non-deterministic) and submits a tiny plan straight to
the control plane -- exactly the bytes the ``aom-swarm`` CLI would write -- then runs the controller
until the workflow completes, exercising the real spawn + wave + pool + publish path over a Modal
Volume. Skipped unless ``AOM_E2E=1`` (spends Modal credits).
"""

from __future__ import annotations

import time

import pytest

from agents_on_modal import sandbox
from agents_on_modal.dynamic import RESERVED, Controller, VolumeControlPlane


def test_dynamic_controller_runs_two_stages(endpoint):
    swarm_id = sandbox.new_id("swarm-")
    shared = f"aom-shared-{swarm_id}"
    control = VolumeControlPlane(shared)
    controller = Controller(control, swarm_id, endpoint, shared, "claude")

    plan = {
        "name": "e2e",
        "stages": [
            {"name": "wave1", "max_concurrent": 1, "agents": [
                {"prompt": "Write the single word ALPHA to /shared/a1.txt, then stop.",
                 "timeout": 600},
                {"prompt": "Write the single word BETA to /shared/a2.txt, then stop.",
                 "timeout": 600}]},
            {"name": "summary", "agents": [
                {"prompt": "Read /shared/a1.txt and /shared/a2.txt and write both words, "
                           "space-separated, to /shared/summary.txt. Then stop.",
                 "timeout": 600}]},
        ],
    }
    control.put_json(f"{RESERVED}/requests/req-e2e.json",
                     {"id": "req-e2e", "verb": "submit", "payload": plan})

    try:
        deadline = time.time() + 1800
        wf = None
        while time.time() < deadline:
            controller.tick()
            wf = control.get_json(f"{RESERVED}/workflows/e2e.json")
            if wf and wf.get("status") in ("done", "cancelled"):
                break
            time.sleep(10)

        assert wf is not None, "controller never published the workflow"
        assert wf["status"] == "done", wf
        assert [s["status"] for s in wf["stages"]] == ["done", "done"]

        summary = b"".join(VolumeControlPlane(shared)._volume().read_file("/summary.txt"))
        assert b"ALPHA" in summary and b"BETA" in summary, summary
    finally:
        for box in sandbox.list_sandboxes("modal", swarm=swarm_id):
            try:
                box.stop()
            except Exception:  # noqa: BLE001
                pass
