#!/usr/bin/env python3
"""Interactive LOCAL session: a harness on *this* machine, against a Modal-served model.

No container -- the harness runs on the host, in a directory you choose (this repo by default).
Everything else (the model endpoint, the harness home, config, instructions) is managed for you.

    # deploy the model once (idempotent), then:
    python examples/local_session.py --harness claude --model qwen3.8-27b

Run `aom setup` first so a Modal proxy token is stored.
"""

from __future__ import annotations

import argparse

from agents_on_modal import models, sandbox


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=("claude", "codex"), default="claude")
    parser.add_argument("--model", default="qwen3.8-27b", help="catalog model name")
    parser.add_argument("--workdir", default=".", help="directory the agent works in (default: cwd)")
    parser.add_argument("--effort", default=None, help="reasoning effort (none..max)")
    parser.add_argument("--deploy", action="store_true",
                        help="deploy + prepare the model first (default: attach to an existing one)")
    args = parser.parse_args()

    model = models.get(args.model)
    endpoint = models.deploy(model) if args.deploy else models.endpoint(model)
    if args.deploy:
        models.wait_ready(endpoint)

    # A local sandbox is just this host; the chosen directory is the agent's workspace.
    box = sandbox.create("local", endpoint=endpoint, workdir=args.workdir)
    try:
        return box.interactive(args.harness, endpoint, effort=args.effort)
    finally:
        box.stop()


if __name__ == "__main__":
    raise SystemExit(main())
