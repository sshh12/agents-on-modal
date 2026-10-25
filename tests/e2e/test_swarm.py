"""End-to-end: a swarm of modal agents -- fan-out, stateless reconnect, and the shared volume.

``Swarm.launch`` deploys + pre-warms the model, so this needs ``HF_TOKEN`` in the environment (the
deploy bakes the HF secret). It fans out ``count`` modal sandboxes on one prompt, each writing a
per-agent artifact to ``/work`` and a collaboration file to the shared ``/shared`` Volume. Skipped
unless ``AOM_E2E=1`` (and skipped without ``HF_TOKEN``).
"""

from __future__ import annotations

import os
import time

import pytest

from agents_on_modal.swarm import Swarm

TERMINAL = ("done", "timeout", "failed", "cancelled")

TASK = (
    "Do two things, then stop. (1) Create /work/DONE.txt containing exactly the word SUCCESS "
    "(no newline). (2) Create a file in the /shared directory named swarm-<random>.txt (pick a "
    "unique random name) containing the word HELLO."
)


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"), reason="Swarm.launch redeploys the model, which needs HF_TOKEN"
)
def test_swarm_fanout_reconnect_shared(model_name):
    swarm = Swarm.launch("claude", model_name, TASK, count=2, timeout=600)
    try:
        deadline = time.time() + 600
        while time.time() < deadline and not all(a.state() in TERMINAL for a in swarm.agents()):
            time.sleep(8)

        # every agent finished and produced its own per-agent artifact
        for agent in swarm.agents():
            assert agent.state() == "done", (agent.id, agent.state())
            done = agent.sandbox.read_file("/work/DONE.txt").decode("utf-8").strip()
            assert done == "SUCCESS", (agent.id, done)

        # stateless reconnect rebuilds the whole swarm from its id alone (dashboard-after-restart)
        again = Swarm.reconnect(swarm.id)
        assert len(again.agents()) == 2

        # agents collaborated through the shared Volume (Modal v2 auto-commits; poll briefly)
        shared: list[str] = []
        for _ in range(8):
            shared = [p for p in swarm.list_shared("/") if p.endswith(".txt")]
            if shared:
                break
            time.sleep(5)
        assert shared, "no files were committed to the shared volume"
    finally:
        swarm.stop()
