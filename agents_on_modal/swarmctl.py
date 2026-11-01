"""``aom-swarm`` — the control CLI a dynamic-swarm orchestrator uses from inside its sandbox.

It never touches Modal. It only reads and writes the control plane under ``/shared/_swarm`` (a plain
directory from inside the sandbox); the local :class:`~agents_on_modal.dynamic.Controller` on the
other side of the Modal Volume turns those files into real sandboxes and publishes state back.

Run as ``aom-swarm <verb>`` (a shim) or ``python3 -m agents_on_modal.swarmctl <verb>``. The control
directory defaults to ``/shared/_swarm`` and can be overridden with ``AOM_SWARM_DIR`` (tests do this).

Verbs: ``submit``, ``spawn``, ``status``, ``trace``, ``wait``, ``stop``, ``resume``. Plans are
linted locally before anything is sent, so mistakes come back immediately instead of silently
failing downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time

from .dynamic import RESERVED, LocalDirControlPlane, parse_plan

TERMINAL = ("done", "timeout", "failed", "cancelled")


def _control() -> LocalDirControlPlane:
    base = os.environ.get("AOM_SWARM_DIR") or f"/shared/{RESERVED}"
    return LocalDirControlPlane(base)


def _reqid() -> str:
    return f"req-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{secrets.token_hex(3)}"


def _submit_request(control: LocalDirControlPlane, verb: str, payload: dict) -> str:
    reqid = _reqid()
    control.put_json(
        f"requests/{reqid}.json", {"id": reqid, "ts": time.time(), "verb": verb, "payload": payload}
    )
    return reqid


def _await_ack(control: LocalDirControlPlane, reqid: str, *, timeout: float, out) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        ack = control.get_json(f"ack/{reqid}.json")
        if ack is not None:
            for w in ack.get("warnings", []):
                print(f"warning: {w}", file=out)
            if ack.get("ok"):
                result = ack.get("result") or {}
                print("ok" + (f": {json.dumps(result)}" if result else ""), file=out)
                return 0
            for e in ack.get("errors", []):
                print(f"error: {e}", file=out)
            return 1
        time.sleep(1.0)
    print(f"timed out after {timeout:g}s waiting for the controller to ack {reqid}; it may still be "
          f"queued behind other work (ack latency climbs under load) -- check 'status'/'trace' before "
          f"resubmitting. A timeout is NOT a failure, and resubmitting creates a NEW request id (the "
          f"controller only dedupes the same id), so a resubmit can double-run the work.", file=out)
    return 2


# --------------------------------------------------------------------------------------------- #
# verbs                                                                                         #
# --------------------------------------------------------------------------------------------- #
def _inline_schema_paths(data, base_dir: str) -> list[str]:
    """Replace each agent's ``"schema": "<path>"`` with that JSON file's contents, in place.

    A path is resolved against ``base_dir`` (the plan file's directory) unless absolute, and each
    file is read once however many agents share it -- so a 100-agent wave can point at one
    ``/shared/schema.json``. Returns one error per unreadable path; the controller only ever sees
    inline objects.
    """

    loaded: dict[str, object] = {}
    errors: list[str] = []
    stages = data.get("stages") if isinstance(data, dict) else None
    for stage in stages if isinstance(stages, list) else []:
        agents = stage.get("agents") if isinstance(stage, dict) else None
        for agent in agents if isinstance(agents, list) else []:
            if not isinstance(agent, dict) or not isinstance(agent.get("schema"), str):
                continue
            path = os.path.join(base_dir, agent["schema"])
            if path not in loaded:
                try:
                    with open(path, encoding="utf-8") as stream:
                        loaded[path] = json.load(stream)
                except (OSError, ValueError) as error:
                    loaded[path] = None
                    errors.append(f"schema file {agent['schema']!r}: {error}")
            if loaded[path] is not None:
                agent["schema"] = loaded[path]
    return errors


def _cmd_submit(args, control, out) -> int:
    raw = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    try:
        data = json.loads(raw)
    except ValueError as error:
        print(f"error: plan is not valid JSON: {error}", file=out)
        return 1
    if args.name:
        data = {**data, "name": args.name}
    base_dir = os.getcwd() if args.file == "-" else os.path.dirname(os.path.abspath(args.file))
    path_errors = _inline_schema_paths(data, base_dir)
    plan, errors, warnings = parse_plan(data)  # instant client-side lint
    errors = path_errors + errors
    for w in warnings:
        print(f"warning: {w}", file=out)
    if errors:
        print(f"plan not submitted: {len(errors)} error(s)", file=out)
        for e in errors:
            print(f"error: {e}", file=out)
        return 1
    reqid = _submit_request(control, "submit", data)
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out)


def _cmd_spawn(args, control, out) -> int:
    agent: dict = {"prompt": args.prompt, "goal": bool(args.goal), "timeout": args.timeout}
    if args.effort:
        agent["effort"] = args.effort
    if args.harness:
        agent["harness"] = args.harness
    if args.schema:
        try:
            agent["schema"] = json.loads(open(args.schema, encoding="utf-8").read())
        except (OSError, ValueError) as error:
            print(f"error: could not read --schema: {error}", file=out)
            return 1
    payload = {"agent": agent}
    if args.workflow:
        payload["workflow"] = args.workflow
    plan, errors, warnings = parse_plan({"stages": [{"name": "spawn", "agents": [agent]}]})
    for w in warnings:
        print(f"warning: {w}", file=out)
    if errors:
        for e in errors:
            print(f"error: {e}", file=out)
        return 1
    reqid = _submit_request(control, "spawn", payload)
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out)


def _load_workflows(control, only: str | None) -> list[dict]:
    out = []
    for name in control.listdir("workflows"):
        if not name.endswith(".json"):
            continue
        wf = control.get_json(f"workflows/{name}")
        if wf and (only is None or wf.get("name") == only):
            out.append(wf)
    return out


def _cmd_status(args, control, out) -> int:
    workflows = _load_workflows(control, args.workflow)
    if args.json:
        print(json.dumps(workflows, indent=2), file=out)
        return 0
    if not workflows:
        print("no workflows yet", file=out)
        return 0
    for wf in workflows:
        print(f"workflow {wf.get('name')}  [{wf.get('status')}]", file=out)
        for stage in wf.get("stages", []):
            agents = stage.get("agents", [])
            print(f"  stage {stage.get('name')}  [{stage.get('status')}]  "
                  f"{len(agents)} agent(s), max_concurrent={stage.get('max_concurrent')}", file=out)
            for a in agents:
                extra = []
                if a.get("goal"):
                    extra.append("goal")
                if a.get("effort"):
                    extra.append(f"effort={a['effort']}")
                if a.get("schema_ok") is not None:
                    extra.append("schema:" + ("ok" if a["schema_ok"] else "FAIL"))
                tail = ("  " + " ".join(extra)) if extra else ""
                print(f"    [{a.get('idx')}] {a.get('id')}  {a.get('state')}{tail}", file=out)
    return 0


def _fmt_event(ev: dict) -> str:
    kind = ev.get("kind", "?")
    d = ev.get("data") or {}
    if kind in ("text", "thinking"):
        return f"[{kind}] {str(d.get('text', ''))[:400]}"
    if kind == "tool_call":
        return f"[tool_call] {d.get('name', '?')} {json.dumps(d.get('input', {}))[:300]}"
    if kind == "tool_result":
        return f"[tool_result] {'error ' if d.get('is_error') else ''}{str(d.get('output', ''))[:300]}"
    if kind == "error":
        return f"[error] {d.get('message', '')}"
    return f"[{kind}] {json.dumps(d)[:200]}"


def _cmd_trace(args, control, out) -> int:
    text = control.read_text(f"traces/{args.agent_id}.jsonl")
    if text is None:
        print(f"no trace for {args.agent_id} yet", file=out)
        return 0
    for line in text.splitlines()[-args.tail:]:
        try:
            print(_fmt_event(json.loads(line)), file=out)
        except ValueError:
            print(line, file=out)
    return 0


def _all_agents(control) -> list[dict]:
    agents = []
    for name in control.listdir("agents"):
        if name.endswith(".json"):
            a = control.get_json(f"agents/{name}")
            if a:
                agents.append(a)
    return agents


def _cmd_wait(args, control, out) -> int:
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        workflows = _load_workflows(control, args.workflow)
        agents = [a for a in _all_agents(control)
                  if args.workflow is None or a.get("workflow") == args.workflow]
        if args.any:
            if any(a.get("state") in TERMINAL for a in agents):
                print("ok: at least one agent finished", file=out)
                return 0
        else:  # --all (default): every known workflow reached a terminal status
            if workflows and all(wf.get("status") in ("done", "cancelled") for wf in workflows):
                print("ok: all workflows finished", file=out)
                return 0
        time.sleep(args.poll)
    # Not a failure: the default timeout keeps one call under the harness's shell timeout, so the
    # caller just waits again. Say where things stand so it doesn't poll `status` in between.
    workflows = _load_workflows(control, args.workflow)
    agents = [a for a in _all_agents(control) if args.workflow is None or a.get("workflow") == args.workflow]
    states = [a.get("state") for a in agents]
    finished = sum(1 for s in states if s in TERMINAL)
    where = (f"{finished}/{len(states)} agents finished across {len(workflows)} workflow(s)"
             if workflows else "no workflow is running yet (a submit may still be pending its ack)")
    print(f"still waiting after {args.timeout:g}s: {where}. Not an error -- run "
          f"`aom-swarm wait` again to keep waiting.", file=out)
    return 2


def _cmd_stop(args, control, out) -> int:
    reqid = _submit_request(control, "stop", {"target": args.target})
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out)


def _cmd_resume(args, control, out) -> int:
    reqid = _submit_request(control, "resume", {"workflow": args.workflow})
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out)


def _cmd_resume_orchestrator(args, control, out) -> int:
    # HOST-side recovery (issue S3): unlike the other verbs this does not go through the control plane
    # -- there is no controller verb that can relaunch the orchestrator, and when it has died there may
    # be nothing driving the swarm at all. It talks to Modal directly (lazy import), so it needs Modal
    # creds; it is meant to run on the host, not from inside a sandbox.
    from .dynamic import resume_orchestrator

    try:
        result = resume_orchestrator(args.swarm, timeout=args.timeout, effort=args.effort, note=args.note)
    except Exception as error:  # noqa: BLE001
        print(f"error: {type(error).__name__}: {error}", file=out)
        return 1
    print("ok: " + json.dumps(result), file=out)
    return 0


# --------------------------------------------------------------------------------------------- #
# argparse                                                                                      #
# --------------------------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aom-swarm", description="Dynamic-swarm control plane CLI.")
    sub = parser.add_subparsers(dest="verb", required=True)

    def ack_arg(p):
        # The "~12 min" acks once seen under load were the orchestrator's /shared mount never
        # showing the controller's writes; the controller now reloads that mount every tick, so an
        # ack lands in seconds. 90 s stays under the harness's 120 s default shell timeout (900 s
        # forced orchestrators to background the call). A timeout is still not a failure (S4).
        p.add_argument("--ack-timeout", type=float,
                       default=float(os.environ.get("AOM_SWARM_ACK_TIMEOUT", "90")),
                       help="seconds to wait for the controller's ack (env AOM_SWARM_ACK_TIMEOUT, default 90)")

    p = sub.add_parser("submit", help="validate and run a plan (waves + pools)")
    p.add_argument("--file", required=True, help="plan JSON file, or - for stdin")
    p.add_argument("--name", help="workflow name (default: the request id)")
    ack_arg(p)
    p.set_defaults(func=_cmd_submit)

    p = sub.add_parser("spawn", help="spawn one subagent")
    p.add_argument("--prompt", required=True)
    p.add_argument("--goal", action="store_true", help="run the subagent under /goal")
    p.add_argument("--timeout", type=float, default=None, help="deadline seconds (required in effect)")
    p.add_argument("--effort")
    p.add_argument("--harness", help="claude or codex (default: the orchestrator's harness)")
    p.add_argument("--schema", help="path to a JSON Schema file (soft output contract)")
    p.add_argument("--workflow", help="attach to / name the workflow")
    ack_arg(p)
    p.set_defaults(func=_cmd_spawn)

    p = sub.add_parser("status", help="show workflows, stages and subagents")
    p.add_argument("--workflow")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("trace", help="print a subagent's recent trace")
    p.add_argument("agent_id")
    p.add_argument("--tail", type=int, default=40)
    p.set_defaults(func=_cmd_trace)

    p = sub.add_parser("wait", help="block until subagents finish")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="until all workflows finish (default)")
    group.add_argument("--any", action="store_true", help="until any one agent finishes")
    p.add_argument("--workflow")
    # 100 s: one call fits the harness's 120 s shell timeout (7200 got the call killed, and the
    # orchestrator fell back to polling `status` every second or two); just call it again.
    p.add_argument("--timeout", type=float, default=100.0,
                   help="seconds to block (default 100, under the shell's timeout); then run it again")
    p.add_argument("--poll", type=float, default=5.0)
    p.set_defaults(func=_cmd_wait)

    p = sub.add_parser("stop", help="pause a workflow (by name) or stop a subagent (by id)")
    p.add_argument("target")
    ack_arg(p)
    p.set_defaults(func=_cmd_stop)

    p = sub.add_parser("resume", help="un-pause a workflow")
    p.add_argument("--workflow", required=True)
    ack_arg(p)
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser("resume-orchestrator",
                       help="HOST-side: relaunch a swarm's orchestrator after it exited (issue S3)")
    p.add_argument("--swarm", required=True, help="swarm id (e.g. swarm-YYYYMMDD-xxxxxx)")
    p.add_argument("--timeout", type=float, default=None, help="orchestrator deadline seconds")
    p.add_argument("--effort", default=None)
    p.add_argument("--note", default=None, help="extra guidance prepended to the resume prompt")
    p.set_defaults(func=_cmd_resume_orchestrator)
    return parser


def main(argv: list[str] | None = None, *, out=None) -> int:
    from .config import ensure_utf8_io

    ensure_utf8_io()  # S10: non-ASCII output must not crash on a cp1252 console
    out = out or sys.stdout
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    # 'spawn' default timeout: fall back here (argparse default None) so the message is friendly.
    if getattr(args, "verb", None) == "spawn" and args.timeout is None:
        from .dynamic import DEFAULT_TIMEOUT

        args.timeout = DEFAULT_TIMEOUT
    return args.func(args, _control(), out)


if __name__ == "__main__":
    raise SystemExit(main())
