#!/usr/bin/env python3
"""Interactive LOCAL session: a harness on *this* machine, against a Modal-served model.

No container -- the harness runs on the host, in a directory you choose (this repo by default).
Everything else (the model endpoint, the harness home, config, instructions) is managed for you.

    # deploy the model once (idempotent), then:
    python examples/local_session.py --harness claude --model qwen3.8-27b

    # anything else goes to the harness: an initial prompt and/or its own flags
    # (after `--`, everything does, even flags named like ours)
    python examples/local_session.py --harness claude --model qwen3.8-27b "fix the failing tests"
    python examples/local_session.py --harness codex --model qwen3.8-27b -- --model x "hi"

Run `aom setup` first so a Modal proxy token is stored.
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
    parser.add_argument("--workdir", default=".", help="directory the agent works in (default: cwd)")
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

    # Always wait for the endpoint to be serving before handing off to the harness. Even on the
    # attach path (no --deploy) the model may be scaled to zero and cold-starting, so the very
    # first request would otherwise hit 503 "no upstreams available" and the harness would spin on
    # its own retries (and give up) while the container boots. wait_ready() polls /health, tolerating
    # the cold-start 503/404, until the model can actually serve.
    print(
        f"[local] waiting for '{args.model}' to be ready "
        "(a cold start can take several minutes)...",
        flush=True,
    )
    models.wait_ready(endpoint)
    print(f"[local] '{args.model}' is ready; starting {args.harness} session.", flush=True)

    # A local sandbox is just this host; the chosen directory is the agent's workspace.
    box = sandbox.create("local", endpoint=endpoint, workdir=args.workdir)
    try:
        return box.interactive(args.harness, endpoint, effort=args.effort, args=_harness_args(extra))
    finally:
        box.stop()


if __name__ == "__main__":
    raise SystemExit(main())
