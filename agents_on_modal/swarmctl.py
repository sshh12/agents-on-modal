"""``aom-swarm`` — the control CLI a dynamic-swarm orchestrator uses from inside its sandbox.

It never touches Modal. It only reads and writes the control plane under ``/shared/_swarm`` (a plain
directory from inside the sandbox); the local :class:`~agents_on_modal.dynamic.Controller` on the
other side of the Modal Volume turns those files into real sandboxes and publishes state back.

Run as ``aom-swarm <verb>`` (a shim) or ``python3 -m agents_on_modal.swarmctl <verb>``. The control
directory defaults to ``/shared/_swarm`` and can be overridden with ``AOM_SWARM_DIR`` (tests do this).

Verbs: ``submit``, ``spawn``, ``status``, ``trace``, ``wait``, ``stop``, ``resume``, and the
host-side ``supervisor`` and ``resume-orchestrator``. Plans are linted locally before anything is
sent, so mistakes come back immediately instead of silently failing downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time

from .dynamic import NOW_FILE, RESERVED, LocalDirControlPlane, parse_plan

TERMINAL = ("done", "timeout", "failed", "cancelled")
#: Schema `spawn` attaches by default when the campaign has written one (relative to the shared
#: root, i.e. ``/shared/schema.json`` in a sandbox).
DEFAULT_SCHEMA = "schema.json"
#: Most note text one `wait` return prints per file; the rest is a pointer to the file.
NOTE_PRINT_CAP = 2000


def _control() -> LocalDirControlPlane:
    base = os.environ.get("AOM_SWARM_DIR") or f"/shared/{RESERVED}"
    return LocalDirControlPlane(base)


def _shared_root(control: LocalDirControlPlane):
    """The shared workspace the control dir lives in (``/shared`` in a sandbox)."""

    return control.base.parent


def _reqid() -> str:
    return f"req-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}-{secrets.token_hex(3)}"


def _submit_request(control: LocalDirControlPlane, verb: str, payload: dict) -> str:
    reqid = _reqid()
    control.put_json(
        f"requests/{reqid}.json", {"id": reqid, "ts": time.time(), "verb": verb, "payload": payload}
    )
    return reqid


def _request_path(control: LocalDirControlPlane, reqid: str) -> str:
    return (control.base / "requests" / f"{reqid}.json").as_posix()


def _await_ack(control: LocalDirControlPlane, reqid: str, *, timeout: float, out,
               hint: str | None = None) -> int:
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
    # The mount is a snapshot: delivery of the ack, not the controller, is what timed out. Point at
    # the on-disk receipt instead of at `status`, and say plainly that this is probably accepted.
    extra = f" ({hint})" if hint else ""
    print(f"timed out after {timeout:g}s waiting for the controller to ack {reqid}{extra}.\n"
          f"The request is on disk at {_request_path(control, reqid)} -- it is very likely already "
          f"accepted (ack delivery lags under load). Do NOT resubmit: a resubmit is a NEW request "
          f"and double-runs the work. Run `aom-swarm wait` (or `status`, which lists it under "
          f"PENDING until the ack lands).", file=out)
    return 2


def _name_of(request: dict) -> str | None:
    payload = request.get("payload") if isinstance(request, dict) else None
    payload = payload if isinstance(payload, dict) else {}
    name = payload.get("name") if request.get("verb") == "submit" else payload.get("workflow")
    return name if isinstance(name, str) and name else None


def _pending_requests(control) -> list[dict]:
    """Requests submitted but not yet acked or claimed by the controller, as far as this mount
    shows (the ack can lag, so these are `pending`, not failed). One listdir per directory, not
    one read per request: a long campaign accumulates hundreds of them."""

    acked = set(control.listdir("ack"))
    handled = set(control.listdir("handled"))
    out = []
    for name in control.listdir("requests"):
        if not name.endswith(".json") or name in acked or name in handled:
            continue
        request = control.get_json(f"requests/{name}")
        if not isinstance(request, dict):
            continue
        payload = request.get("payload") if isinstance(request.get("payload"), dict) else {}
        if request.get("verb") == "spawn":
            count = 1
        elif request.get("verb") == "submit":
            stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
            count = sum(len(s.get("agents") or []) for s in stages if isinstance(s, dict))
        else:
            count = 0
        try:
            age = max(0.0, time.time() - float(request.get("ts") or time.time()))
        except (TypeError, ValueError):
            age = 0.0
        out.append({"id": name[: -len(".json")], "verb": request.get("verb"),
                    "name": _name_of(request), "agents": count, "age": age,
                    "payload": payload})
    return out


def _fmt_age(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 120 else f"{seconds / 60:.0f} min"


def _free_name(taken: set[str], wanted: str) -> str:
    i = 2
    while f"{wanted}-{i}" in taken:
        i += 1
    return f"{wanted}-{i}"


def _check_submit_name(control, wanted: str | None, out) -> bool:
    """Refuse a `submit` whose workflow name is already pending or running.

    Nearly every such collision is a resubmit of a submit whose ack was slow ("i1 was submitted
    five times"): the first one is queued or running already, so a second copy double-runs the
    whole wave. A deliberate re-run just takes a new name, which the message suggests."""

    if not wanted:
        return True
    all_pending = _pending_requests(control)
    pending = [r for r in all_pending if r["verb"] == "submit" and r["name"] == wanted]
    existing = {n[: -len(".json")] for n in control.listdir("workflows") if n.endswith(".json")}
    if not pending and wanted not in existing:
        print(f"workflow name: {wanted}", file=out)
        return True
    taken = existing | {r["name"] for r in all_pending if r["name"]}
    if pending:
        req = pending[0]
        why = (f"a submit named {wanted!r} is already pending ({req['id']}, "
               f"{_fmt_age(req['age'])} old) -- it is very likely accepted, its ack is just slow")
    else:
        why = f"workflow {wanted!r} already exists (see `aom-swarm status --workflow {wanted}`)"
    print(f"error: not submitted: {why}. Run `aom-swarm wait` instead of resubmitting. To run the "
          f"plan AGAIN on purpose, pass a new name, e.g. --name {_free_name(taken, wanted)}.",
          file=out)
    return False


def _check_spawn_workflow(control, wanted: str | None, agent: dict, out) -> bool:
    """Whether a `spawn --workflow` may go out. An identical spawn still pending is refused (a
    resubmit after a slow ack); a name already used is kept -- the controller files it under a
    suffixed name and says so -- since retrying a lane under its own name is the normal use."""

    if not wanted:
        return True
    for req in _pending_requests(control):
        if (req["verb"] == "spawn" and req["name"] == wanted
                and (req["payload"].get("agent") or {}).get("prompt") == agent.get("prompt")):
            print(f"error: not spawned: an identical spawn for workflow {wanted!r} is already "
                  f"pending ({req['id']}, {_fmt_age(req['age'])} old) -- it is very likely "
                  f"accepted. Run `aom-swarm wait` instead of respawning.", file=out)
            return False
    if f"{wanted}.json" in control.listdir("workflows"):
        print(f"note: workflow {wanted!r} exists; the controller will file this spawn under a "
              f"suffixed name (printed in the ack).", file=out)
    return True


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
    if not isinstance(data, dict):
        print("error: plan must be a JSON object with a 'stages' list", file=out)
        return 1
    if args.name:
        data = {**data, "name": args.name}
    name = data.get("name") if isinstance(data.get("name"), str) else None
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
    if not _check_submit_name(control, name, out):
        return 1
    reqid = _submit_request(control, "submit", data)
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out,
                      hint=f"workflow {name}" if name else None)


def _cmd_spawn(args, control, out) -> int:
    agent: dict = {"prompt": args.prompt, "goal": bool(args.goal), "timeout": args.timeout}
    if args.effort:
        agent["effort"] = args.effort
    if args.harness:
        agent["harness"] = args.harness
    if args.lane:
        agent["lane"] = args.lane
    if args.report_dir:
        agent["report_dir"] = args.report_dir
    schema = args.schema
    default_schema = _shared_root(control) / DEFAULT_SCHEMA
    if schema is None and not args.no_schema and default_schema.is_file():
        schema = str(default_schema)
        print(f"schema: {default_schema.as_posix()} (the campaign default; --no-schema to skip)",
              file=out)
    if schema:
        try:
            agent["schema"] = json.loads(open(schema, encoding="utf-8").read())
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
    if not _check_spawn_workflow(control, args.workflow, agent, out):
        return 1
    reqid = _submit_request(control, "spawn", payload)
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out,
                      hint=f"workflow {args.workflow}" if args.workflow else None)


def _load_workflows(control, only: str | None) -> list[dict]:
    out = []
    for name in control.listdir("workflows"):
        if not name.endswith(".json"):
            continue
        wf = control.get_json(f"workflows/{name}")
        if not wf:
            # Caught mid-update (the volume syncs files in place): report it instead of silently
            # dropping it -- a vanished workflow made `status` print [] and watch loops exit early,
            # and `wait --all` must not count it as done.
            wf = {"name": name[: -len(".json")], "status": "unreadable", "stages": []}
        if only is None or wf.get("name") == only:
            out.append(wf)
    return out


def _controller_now(control) -> float | None:
    """The controller's clock as of this mount's last reload (``_swarm/now.json``). Ages of
    controller-stamped fields must be measured against it: a sandbox clock can be hours off."""

    now = control.get_json(NOW_FILE)
    try:
        return float(now["ts"]) if isinstance(now, dict) else None
    except (KeyError, TypeError, ValueError):
        return None


#: A non-terminal agent whose record the controller has not refreshed for this long is flagged.
STUCK_AFTER = 300.0


def _cmd_status(args, control, out) -> int:
    workflows = _load_workflows(control, args.workflow)
    if args.json:
        print(json.dumps(workflows, indent=2), file=out)
        return 0
    pending = _pending_requests(control)
    for req in pending:
        print(f"PENDING {req['verb']} {req['name'] or '?'}  {req['agents']} agent(s), "
              f"{_fmt_age(req['age'])} old  ({req['id']})", file=out)
    if pending:
        print("  pending = not yet acked; it is very likely accepted. Do not resubmit.", file=out)
    if not workflows:
        print("no workflows yet" if not pending else "no workflow visible yet (see PENDING)", file=out)
        return 0
    if args.workflow is None:
        total = sum(len(s.get("agents", [])) for wf in workflows for s in wf.get("stages", []))
        print(f"{len(workflows)} workflow(s), {total} agent(s) started", file=out)
    now = _controller_now(control)
    for wf in workflows:
        agents_in = [a for s in wf.get("stages", []) for a in s.get("agents", [])]
        live = sum(1 for a in agents_in if a.get("state") not in TERMINAL)
        print(f"workflow {wf.get('name')}  [{wf.get('status')}]  "
              f"{live} running of {len(agents_in)} started", file=out)
        for stage in wf.get("stages", []):
            agents = stage.get("agents", [])
            planned = stage.get("planned")
            count = f"{len(agents)}/{planned}" if planned is not None else f"{len(agents)}"
            print(f"  stage {stage.get('name')}  [{stage.get('status')}]  "
                  f"{count} agent(s) started, max_concurrent={stage.get('max_concurrent')}",
                  file=out)
            for a in agents:
                extra = []
                if a.get("goal"):
                    extra.append("goal")
                if a.get("effort"):
                    extra.append(f"effort={a['effort']}")
                if a.get("lane"):
                    extra.append(f"lane={a['lane']}")
                if a.get("schema_ok") is not None:
                    extra.append("schema:" + ("ok" if a["schema_ok"] else "FAIL"))
                if now is not None and a.get("state") not in TERMINAL and a.get("updated_at"):
                    stale = now - float(a["updated_at"])
                    if stale > STUCK_AFTER:
                        extra.append(f"STUCK {stale / 60:.0f}m (no status read)")
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


#: The host-side `aom-swarm supervisor` writes here; it is the one note that outranks the plan.
SUPERVISOR_NOTE = "NOTE-supervisor.md"


def _notes_seen_path(control):
    """Where this sandbox remembers how much of each note it has printed. Local (not under
    ``/shared``): a replacement orchestrator is a new sandbox and must see every note again."""

    import hashlib
    import tempfile
    from pathlib import Path

    key = hashlib.sha1(str(control.base).encode("utf-8")).hexdigest()[:10]
    return Path(os.environ.get("AOM_SWARM_NOTES_SEEN")
                or Path(tempfile.gettempdir()) / f"aom-swarm-notes-{key}.json")


def _print_new_notes(control, out) -> None:
    """Print what was added to ``/shared/NOTE*`` since this sandbox last printed it. A note has to
    be spliced into the loop the orchestrator is already in (every `wait` return), not left for a
    re-read it may never do mid-turn."""

    seen_path = _notes_seen_path(control)
    try:
        seen = json.loads(seen_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    if not isinstance(seen, dict):
        seen = {}
    try:
        notes = sorted(p for p in _shared_root(control).glob("NOTE*") if p.is_file())
    except OSError:
        return
    changed = False
    for path in notes:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        try:
            offset = int(seen.get(path.name) or 0)
        except (TypeError, ValueError):
            offset = 0
        if size == offset:
            continue
        rewritten = size < offset  # replaced wholesale: show it from the top
        start = 0 if rewritten else offset
        try:
            with open(path, "rb") as stream:
                stream.seek(start)
                data = stream.read()
        except OSError:
            continue
        text = data.decode("utf-8", "replace").strip()
        seen[path.name] = start + len(data)
        changed = True
        if not text:
            continue
        label = ("supervisor note (outranks the plan)" if path.name == SUPERVISOR_NOTE
                 else f"note {path.name}")
        print(f"--- {label}{' (rewritten)' if rewritten else ''}, new since your last wait:",
              file=out)
        if len(text) > NOTE_PRINT_CAP:
            print(text[:NOTE_PRINT_CAP], file=out)
            print(f"... ({len(text) - NOTE_PRINT_CAP} more chars: read {path.as_posix()})", file=out)
        else:
            print(text, file=out)
    if changed:
        try:
            seen_path.parent.mkdir(parents=True, exist_ok=True)
            seen_path.write_text(json.dumps(seen), encoding="utf-8")
        except OSError:
            pass


def _agents_of(workflows: list[dict]) -> list[dict]:
    """Every started agent in ``workflows`` (their records ride in the workflow files, so `wait`
    needs no per-agent read: 100+ file reads on the mount every poll)."""

    return [a for wf in workflows for s in wf.get("stages", []) for a in s.get("agents", [])]


def _cmd_wait(args, control, out) -> int:
    _print_new_notes(control, out)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        workflows = _load_workflows(control, args.workflow)
        agents = _agents_of(workflows)
        if args.any:
            if any(a.get("state") in TERMINAL for a in agents):
                _print_new_notes(control, out)
                print("ok: at least one agent finished", file=out)
                return 0
        else:  # --all (default): every known workflow reached a terminal status
            if workflows and all(wf.get("status") in ("done", "cancelled") for wf in workflows):
                _print_new_notes(control, out)
                print("ok: all workflows finished", file=out)
                return 0
        time.sleep(args.poll)
    # Not a failure: the default timeout keeps one call under the harness's shell timeout, so the
    # caller just waits again. Say where things stand so it doesn't poll `status` in between.
    workflows = _load_workflows(control, args.workflow)
    states = [a.get("state") for a in _agents_of(workflows)]
    finished = sum(1 for s in states if s in TERMINAL)
    where = (f"{finished}/{len(states)} agents finished across {len(workflows)} workflow(s)"
             if workflows else "no workflow is running yet (a submit may still be pending its ack)")
    print(f"still waiting after {args.timeout:g}s: {where}. Not an error -- run "
          f"`aom-swarm wait` again to keep waiting.", file=out)
    _print_new_notes(control, out)
    return 2


def _cmd_stop(args, control, out) -> int:
    # Say what the target covers before acting: an unscoped or mistyped stop is how a sweep of
    # stragglers once paused the wrong workflow.
    target = args.target
    workflows = _load_workflows(control, None)
    named = [wf for wf in workflows if wf.get("name") == target]
    if named:
        agents = _agents_of(named)
        live = sum(1 for a in agents if a.get("state") not in TERMINAL)
        print(f"pausing workflow {target}: {len(agents)} agent(s) started, {live} running. No new "
              f"agent of it starts; the running ones keep going (stop them by id, or `resume`).",
              file=out)
    else:
        for wf in workflows:
            hit = next((a for a in _agents_of([wf]) if a.get("id") == target), None)
            if hit is not None:
                state = hit.get("state")
                print(f"stopping agent {target} (workflow {wf.get('name')}, {state})"
                      + (" -- it has already finished; its result is kept." if state in TERMINAL
                         else ""), file=out)
                break
    reqid = _submit_request(control, "stop", {"target": target})
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out)


def _cmd_resume(args, control, out) -> int:
    reqid = _submit_request(control, "resume", {"workflow": args.workflow})
    return _await_ack(control, reqid, timeout=args.ack_timeout, out=out)


def _cmd_supervisor(args, control, out) -> int:
    """HOST-side: append a supervisor note to ``/shared/NOTE-supervisor.md``.

    The note is the only way to correct a live run without restarting it. The orchestrator now sees
    new lines on every `wait` return and every digest.
    """

    from .dynamic import VolumeControlPlane

    swarm = args.swarm or os.environ.get("AOM_SWARM") or ""
    if not swarm:
        print("error: --swarm (or AOM_SWARM) is required", file=out)
        return 1
    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    if not text.strip():
        print("error: empty note", file=out)
        return 1
    # create=False: a mistyped swarm id must not mint an empty volume and "deliver" the note there.
    cp = VolumeControlPlane(f"aom-shared-{swarm}", create=False)
    if cp.get_json(f"{RESERVED}/meta.json") is None:
        print(f"error: no swarm {swarm!r} (no {RESERVED}/meta.json on aom-shared-{swarm})", file=out)
        return 1
    existing = cp.read_text(SUPERVISOR_NOTE) or "# Supervisor notes (outrank the plan)\n"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cp.put_text(SUPERVISOR_NOTE, existing.rstrip() + f"\n\n## {stamp}\n{text.strip()}\n")
    print(f"ok: appended {len(text.strip())} chars to {SUPERVISOR_NOTE} on aom-shared-{swarm}; the "
          f"orchestrator sees it on its next `aom-swarm wait` return", file=out)
    return 0


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
        # 100 s, like `wait`: one call must fit the harness's 120 s shell timeout, or the shell
        # kills it before the "do not resubmit" message prints. Acks used to take 90-105 s under a
        # spawn wave because requests were only read between ticks; the controller now reads them
        # while a wave is being created, and a resubmit after a slow ack is refused (same name).
        p.add_argument("--ack-timeout", type=float,
                       default=float(os.environ.get("AOM_SWARM_ACK_TIMEOUT", "100")),
                       help="seconds to wait for the controller's ack (env AOM_SWARM_ACK_TIMEOUT, "
                            "default 100)")

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
    p.add_argument("--schema", help="path to a JSON Schema file (soft output contract; default: "
                                    "/shared/schema.json when it exists)")
    p.add_argument("--no-schema", action="store_true",
                   help="do not attach the default /shared/schema.json (setup/helper agents)")
    p.add_argument("--lane", help="the lane token this agent is recorded under")
    p.add_argument("--report-dir", help="absolute /shared/... directory the agent writes its report to")
    p.add_argument("--workflow", help="name the one-agent workflow this spawn creates (a name "
                                      "already in use gets a suffix, printed in the ack)")
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

    p = sub.add_parser("supervisor", help="HOST-side: append a note to the run's NOTE-supervisor.md")
    p.add_argument("--swarm", help="swarm id (default: $AOM_SWARM)")
    p.add_argument("--file", required=True, help="note file, or - for stdin")
    p.set_defaults(func=_cmd_supervisor)

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
