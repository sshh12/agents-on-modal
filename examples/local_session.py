#!/usr/bin/env python3
"""Interactive LOCAL session: a harness on *this* machine, against a Modal-served model.

No container -- the harness runs on the host, in a directory you choose (this repo by default).
Everything else (the model endpoint, the harness home, config, instructions) is managed for you.

    # deploys the model first if it isn't deployed (a live one is attached to, never restarted)
    python examples/local_session.py --harness claude --model qwen3.8-27b

    # anything else goes to the harness: an initial prompt and/or its own flags
    # (after `--`, everything does, even flags named like ours)
    python examples/local_session.py --harness claude --model qwen3.8-27b "fix the failing tests"
    python examples/local_session.py --harness codex --model qwen3.8-27b -- --model x "hi"

Setup is automatic (``examples/bootstrap.py``): it loads the repo ``.env`` (``HF_TOKEN`` for a
deploy), re-launches on the repo ``.venv`` if this Python can't run aom, and runs the Modal login /
proxy-token mint if missing -- so plain ``python examples/local_session.py ...`` works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for agents_on_modal
sys.path.insert(0, str(Path(__file__).resolve().parent))  # examples/, for the shared bootstrap

import bootstrap  # noqa: E402 - stdlib-only module shared by the examples

if __name__ == "__main__":
    bootstrap.prepare(interactive=True)  # .env, .venv, Modal login -- before the aom imports below

from agents_on_modal import models, sandbox  # noqa: E402


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
    # A model that isn't deployed is always deployed now; --deploy is kept so old command lines
    # parse (unrecognised args go to the harness, which would reject it).
    parser.add_argument("--deploy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-containers", type=int, default=1,
                        help="the endpoint's GPU container cap to ask for: a first deploy uses it, "
                             "a live endpoint is only raised to it, never restarted")
    # Anything we don't recognise (an initial prompt, harness flags) is passed to the harness.
    args, extra = parser.parse_known_args()

    # Attach if deployed, deploy (+ prepare weights) if not -- e.g. after `aom models stop`.
    model = models.get(args.model)
    endpoint = models.ensure_deployed(model, max_containers=args.max_containers)

    # Always wait for the endpoint to be serving before handing off to the harness. The model may
    # be scaled to zero and cold-starting, so the very first request would otherwise hit 503 "no
    # upstreams available" and the harness would spin on its own retries (and give up) while the
    # container boots. wait_ready() polls /health, tolerating the cold-start 503/404, until the
    # model can actually serve.
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
