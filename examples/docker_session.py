#!/usr/bin/env python3
"""Interactive DOCKER session: a harness in a local container, batteries included.

The container is built from the aom image (dev/cyber toolchain, headless Chromium, Playwright MCP,
agent-browser, gh, ...). The agent has internet but no access to your LAN or host; the model token
is injected by the egress proxy, so the agent never sees it. Everything is managed for you.

    python examples/docker_session.py --harness codex --model qwen3.8-27b
    python examples/docker_session.py --harness claude --model qwen3.8-27b \\
        --proxy socks5h://user:pass@residential.example:1080

Needs Docker running locally and a stored Modal proxy token (`aom setup`).
"""

from __future__ import annotations

import argparse

from agents_on_modal import models, sandbox


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=("claude", "codex"), default="claude")
    parser.add_argument("--model", default="qwen3.8-27b", help="catalog model name")
    parser.add_argument("--proxy", default=None,
                        help="upstream proxy URL for all agent egress ({session} supported)")
    parser.add_argument("--effort", default=None, help="reasoning effort (none..max)")
    parser.add_argument("--deploy", action="store_true",
                        help="deploy + prepare the model first (default: attach to an existing one)")
    args = parser.parse_args()

    model = models.get(args.model)
    endpoint = models.deploy(model) if args.deploy else models.endpoint(model)
    if args.deploy:
        models.wait_ready(endpoint)

    box = sandbox.create("docker", endpoint=endpoint, proxy=args.proxy)
    try:
        return box.interactive(args.harness, endpoint, effort=args.effort)
    finally:
        box.stop()


if __name__ == "__main__":
    raise SystemExit(main())
