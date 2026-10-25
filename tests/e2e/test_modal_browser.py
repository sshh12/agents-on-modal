"""End-to-end: the batteries browser (Playwright MCP + headless Chromium) in the modal sandbox.

Verifies the browser/dev-tools are low-friction: the agent is handed the Playwright MCP server and a
headless Chromium with zero extra setup, navigates a real page, and reports what it read back. Skipped
unless ``AOM_E2E=1``.
"""

from __future__ import annotations

import os

import pytest

from agents_on_modal import sandbox

# The browser recipe is validated with Claude (Playwright MCP tools); other harnesses can opt in.
HARNESSES = [h for h in os.environ.get("AOM_E2E_BROWSER_HARNESSES", "claude").split(",") if h]

BROWSER_TASK = (
    "Use the Playwright browser tools to open https://example.com , read the main heading (the H1 "
    "text) on the page, and write that heading text to /work/browser.txt. Then stop."
)


@pytest.mark.parametrize("harness", HARNESSES)
def test_modal_browser(endpoint, harness):
    box = sandbox.create("modal", endpoint=endpoint)
    try:
        run = box.start(harness, endpoint, BROWSER_TASK, timeout=900)
        for _ in run.events(follow=True, poll=3.0):
            pass
        status = run.wait(timeout=60)

        assert status["state"] == "done", status
        content = box.read_file("/work/browser.txt").decode("utf-8")
        assert "example domain" in content.lower(), content
    finally:
        box.stop()
