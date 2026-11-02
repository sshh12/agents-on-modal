#!/usr/bin/env python3
"""Interactive DOCKER session: a harness in a local container, batteries included.

The container is built from the aom image (dev/cyber toolchain, headless Chromium, Playwright MCP,
agent-browser, gh, ...). The agent has internet but no access to your LAN or host; the model token
is injected by the egress proxy, so the agent never sees it. Everything is managed for you.

    python examples/docker_session.py --harness codex --model qwen3.8-27b
    python examples/docker_session.py --harness claude --model qwen3.8-27b \\
        --proxy socks5h://user:pass@residential.example:1080

    # anything else goes to the harness: an initial prompt and/or its own flags
    # (after `--`, everything does, even flags named like ours)
    python examples/docker_session.py --harness claude --model qwen3.8-27b "audit this repo"

Needs Docker running locally and a stored Modal proxy token (`aom setup`).
"""

from __future__ import annotations

import argparse

from agents_on_modal import models, sandbox


def _harness_args(extra: list[str]) -> list[str]:
    """Unrecognised args go to the harness verbatim; the first ``--`` only ends our options."""

    if "--" in extra:
        extra = list(extra)
        extra.remove("--")
    return extra


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] [harness args...]",
    )
    parser.add_argument("--harness", choices=("claude", "codex"), default="claude")
    parser.add_argument("--model", default="qwen3.8-27b", help="catalog model name")
    parser.add_argument("--proxy", default=None,
                        help="upstream proxy URL for all agent egress ({session} supported)")
    parser.add_argument("--effort", default=None, help="reasoning effort (none..max)")
    parser.add_argument("--deploy", action="store_true",
                        help="deploy + prepare the model first if it isn't deployed "
                             "(default: attach to an existing one)")
    parser.add_argument("--max-containers", type=int, default=1,
                        help="with --deploy: the endpoint's GPU container cap to ask for "
                             "(a live endpoint is only raised to it, never restarted)")
    # Anything we don't recognise (an initial prompt, harness flags) is passed to the harness.
    args, extra = parser.parse_known_args()

    model = models.get(args.model)
    endpoint = (models.ensure_deployed(model, max_containers=args.max_containers) if args.deploy
                else models.endpoint(model))
    if args.deploy:
        models.wait_ready(endpoint)

    box = sandbox.create("docker", endpoint=endpoint, proxy=args.proxy)
    try:
        return box.interactive(args.harness, endpoint, effort=args.effort, args=_harness_args(extra))
    finally:
        box.stop()


if __name__ == "__main__":
    raise SystemExit(main())
