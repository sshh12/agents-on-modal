"""Offline tests for dynamic swarms: plan model, pure scheduler, control plane, the controller
(pool + waves + stop/resume + schema), the ``aom-swarm`` CLI, and ``DynamicSwarm.launch``.

No Modal. ``sandbox``/``models`` seams are faked or monkeypatched exactly like ``test_swarm.py``.
"""

from __future__ import annotations

import io
import itertools
import json
import threading
import time
from pathlib import Path

import pytest

import agents_on_modal.dynamic as dynamic
import agents_on_modal.swarmctl as swarmctl
from agents_on_modal import models, sandbox
from agents_on_modal.dynamic import (
    DEFAULT_TIMEOUT,
    Controller,
    DynamicSwarm,
    LocalDirControlPlane,
    Plan,
    PlanError,
    Stage,
    active_stage_index,
    build_orchestrator_instructions,
    parse_plan,
    stage_complete,
    stage_next_indices,
    validate_against_schema,
)
from agents_on_modal.models import lifecycle as model_ops

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "swarm"


# --------------------------------------------------------------------------------------------- #
# fakes                                                                                        #
# --------------------------------------------------------------------------------------------- #
class FakeModel:
    def __init__(self, name="m"):
        self.name = name
        self.scaling = models.Scaling()


class FakeEndpoint:
    def __init__(self):
        self.model = FakeModel("qwen3.8-27b")
        self.url = "https://x.modal.direct"
        self.token = "wk-a.ws-b"


class FakeRun:
    def __init__(self, run_id="run-x", state="running"):
        self.run_id = run_id
        self.state = state

    def status(self):
        return {"state": self.state}

    def read_trace(self, offset=0):
        return [], offset


class FakeBox:
    _ids = itertools.count(1)  # the controller creates boxes from several threads at once

    def __init__(self):
        self.id = f"aom-{next(FakeBox._ids):04d}"
        self.started = None
        self.stopped = False
        self._run = FakeRun(run_id=f"{self.id}-run")

    def start(self, harness, endpoint, prompt, **kw):
        self.started = {"harness": harness, "prompt": prompt, **kw}
        return self._run

    def run(self, run_id):
        return self._run

    def runs(self):
        return [self._run]

    def write_file(self, path, data):  # for the CLI shim install
        pass

    def exec(self, argv, **kw):
        pass

    def stop(self):
        self.stopped = True


class FakeSandboxes:
    """Stand-in for the ``sandbox`` module used by the controller."""

    def __init__(self):
        self.created: list[FakeBox] = []

    def create(self, name, **opts):
        box = FakeBox()
        box.create_opts = {"name": name, **opts}
        self.created.append(box)
        return box

    def get(self, name, sid):
        for b in self.created:
            if b.id == sid:
                return b
        raise KeyError(sid)

    def list_sandboxes(self, name, **filters):
        return list(self.created)


def _controller(tmp_path, sb=None):
    control = LocalDirControlPlane(str(tmp_path))
    sb = sb or FakeSandboxes()
    return Controller(control, "sw", FakeEndpoint(), "aom-shared-sw", "claude", sandboxes=sb), control, sb


# --------------------------------------------------------------------------------------------- #
# plan model + lint                                                                            #
# --------------------------------------------------------------------------------------------- #
def test_parse_plan_valid_fills_defaults():
    plan, errors, warnings = parse_plan(
        {"stages": [{"name": "w1", "agents": [{"prompt": "do it"}]}]}
    )
    assert errors == [] and warnings == []
    assert isinstance(plan, Plan)
    agent = plan.stages[0].agents[0]
    assert agent.timeout == DEFAULT_TIMEOUT and agent.goal is False and agent.effort is None
    # max_concurrent defaults to "all at once"
    assert plan.stages[0].max_concurrent == 1


def test_parse_plan_missing_prompt_is_error():
    plan, errors, _ = parse_plan({"stages": [{"name": "s", "agents": [{"goal": True}]}]})
    assert plan is None
    assert any("prompt" in e for e in errors)


def test_parse_plan_unknown_effort_warns_and_drops():
    plan, errors, warnings = parse_plan(
        {"stages": [{"agents": [{"prompt": "x", "effort": "turbo"}]}]}
    )
    assert errors == []
    assert plan.stages[0].agents[0].effort is None
    assert any("effort" in w for w in warnings)


def test_parse_plan_bad_timeout_and_empty_stage():
    _, errors, _ = parse_plan(
        {"stages": [{"name": "s", "agents": [{"prompt": "x", "timeout": -5}]}]}
    )
    assert any("timeout" in e for e in errors)
    _, errors2, _ = parse_plan({"stages": [{"name": "empty", "agents": []}]})
    assert any("agents" in e for e in errors2)
    _, errors3, _ = parse_plan({"stages": []})
    assert errors3  # no stages


def test_plan_from_dict_raises_on_error():
    with pytest.raises(PlanError):
        Plan.from_dict({"stages": [{"agents": [{}]}]})


# --------------------------------------------------------------------------------------------- #
# pure scheduler                                                                               #
# --------------------------------------------------------------------------------------------- #
def test_stage_next_indices_respects_max_concurrent():
    stage = Stage("s", [dynamic.AgentSpec(prompt=p) for p in "abcd"], max_concurrent=2)
    assert stage_next_indices(stage, started=set(), finished=set()) == [0, 1]
    # one running, one finished -> one free slot, next pending is 2
    assert stage_next_indices(stage, started={0, 1}, finished={1}) == [2]
    # all slots busy
    assert stage_next_indices(stage, started={0, 1}, finished=set()) == []


def test_stage_complete_and_active_stage_order():
    s1 = Stage("w1", [dynamic.AgentSpec(prompt="a"), dynamic.AgentSpec(prompt="b")])
    s2 = Stage("w2", [dynamic.AgentSpec(prompt="c")])
    plan = Plan([s1, s2])
    assert not stage_complete(s1, {0})
    assert stage_complete(s1, {0, 1})
    assert active_stage_index(plan, [set(), set()]) == 0
    assert active_stage_index(plan, [{0, 1}, set()]) == 1  # wave 2 only after wave 1
    assert active_stage_index(plan, [{0, 1}, {0}]) is None


# --------------------------------------------------------------------------------------------- #
# control plane + schema                                                                       #
# --------------------------------------------------------------------------------------------- #
def test_local_control_plane_roundtrip(tmp_path):
    cp = LocalDirControlPlane(str(tmp_path))
    cp.put_json("a/b.json", {"x": 1})
    assert cp.get_json("a/b.json") == {"x": 1}
    assert cp.listdir("a") == ["b.json"]
    assert cp.get_json("missing.json") is None
    assert cp.listdir("nope") == []


def test_schema_validator():
    schema = {"type": "object", "required": ["name"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    assert validate_against_schema({"name": "x", "age": 3}, schema) == []
    problems = validate_against_schema({"age": "no"}, schema)
    assert any("missing required property 'name'" in p for p in problems)
    assert any("age" in p for p in problems)


# --------------------------------------------------------------------------------------------- #
# controller: waves + pool                                                                     #
# --------------------------------------------------------------------------------------------- #
def test_controller_submit_runs_pool_then_wave(tmp_path):
    c, control, sb = _controller(tmp_path)
    plan = {"name": "wf", "stages": [
        {"name": "w1", "max_concurrent": 2, "agents": [
            {"prompt": "a", "goal": True, "timeout": 123, "effort": "high"},
            {"prompt": "b"}, {"prompt": "c"}]},
        {"name": "w2", "agents": [{"prompt": "summary"}]}]}
    control.put_json("_swarm/requests/req-1.json", {"id": "req-1", "verb": "submit", "payload": plan})

    c.tick()
    assert control.get_json("_swarm/ack/req-1.json")["ok"] is True
    assert len(sb.created) == 2  # pool cap
    # spawns run concurrently, so creation order is not index order: look agents up by idx
    a0 = next(b.started for b in sb.created if b.started["meta"]["idx"] == 0)
    assert a0["goal"] is True and a0["timeout"] == 123 and a0["effort"] == "high"
    assert a0["meta"] == {"swarm_id": "sw", "workflow": "wf", "stage": "w1", "idx": 0, "role": "subagent"}

    for b in sb.created:
        b._run.state = "done"
    c.tick()
    assert len(sb.created) == 3  # third of w1 starts as a slot frees

    sb.created[2]._run.state = "done"
    c.tick()
    assert len(sb.created) == 4  # w1 complete -> w2 starts
    assert sb.created[3].started["meta"]["stage"] == "w2"

    sb.created[3]._run.state = "done"
    c.tick()
    wf = control.get_json("_swarm/workflows/wf.json")
    assert wf["status"] == "done"
    assert [s["status"] for s in wf["stages"]] == ["done", "done"]


def test_controller_spawn_one_off(tmp_path):
    c, control, sb = _controller(tmp_path)
    control.put_json("_swarm/requests/r.json",
                     {"id": "r", "verb": "spawn",
                      "payload": {"workflow": "adhoc", "agent": {"prompt": "hi", "timeout": 60}}})
    c.tick()
    assert control.get_json("_swarm/ack/r.json")["ok"] is True
    assert len(sb.created) == 1 and sb.created[0].started["timeout"] == 60


def test_controller_propagates_proxy_and_note_to_subagents(tmp_path):
    # the swarm's proxy mode must reach EVERY fan-out subagent, not just the orchestrator.
    PROXY = "socks5h://u-session-{session}:p@gate.decodo.com:10000"
    NOTE = ("## Residential proxies (optional — browser/web ONLY)\n"
            "SOCKS5 endpoints: socks5h://u-session-1:p@gate.decodo.com:10000\nWHEN: ...\nHOW: ...")
    control = LocalDirControlPlane(str(tmp_path))
    sb = FakeSandboxes()
    c = Controller(control, "sw", FakeEndpoint(), "aom-shared-sw", "claude",
                   proxy=PROXY, proxy_note=NOTE, sandboxes=sb)
    plan = {"name": "wf", "stages": [{"name": "s", "agents": [{"prompt": "fetch a page"}]}]}
    control.put_json("_swarm/requests/r.json", {"id": "r", "verb": "submit", "payload": plan})
    c.tick()
    assert len(sb.created) == 1
    box = sb.created[0]
    # always-mode: the subagent sandbox is created WITH the proxy (its relay -> its own sticky IP)
    assert box.create_opts["proxy"] == PROXY
    # agent-opt-in mode: the note is injected into the subagent's own prompt
    assert NOTE in box.started["prompt"] and "fetch a page" in box.started["prompt"]


def test_controller_stop_pauses_workflow_and_stops_agent(tmp_path):
    c, control, sb = _controller(tmp_path)
    control.put_json("_swarm/requests/s1.json", {"id": "s1", "verb": "submit", "payload": {
        "name": "wf", "stages": [{"name": "w1", "max_concurrent": 1,
                                  "agents": [{"prompt": "a"}, {"prompt": "b"}]}]}})
    c.tick()
    assert len(sb.created) == 1
    # pause the workflow -> no new agents even when the running one finishes
    control.put_json("_swarm/requests/s2.json", {"id": "s2", "verb": "stop", "payload": {"target": "wf"}})
    sb.created[0]._run.state = "done"
    c.tick()
    assert control.get_json("_swarm/ack/s2.json")["result"]["status"] == "paused"
    assert len(sb.created) == 1  # paused: no second agent
    assert control.get_json("_swarm/workflows/wf.json")["status"] == "paused"
    # resume -> the next agent starts
    control.put_json("_swarm/requests/s3.json", {"id": "s3", "verb": "resume", "payload": {"workflow": "wf"}})
    c.tick()
    assert len(sb.created) == 2
    # stop an individual agent id
    target = sb.created[1].id
    control.put_json("_swarm/requests/s4.json", {"id": "s4", "verb": "stop", "payload": {"target": target}})
    c.tick()
    assert sb.created[1].stopped is True


def test_controller_schema_soft_contract(tmp_path):
    c, control, sb = _controller(tmp_path)
    schema = {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}}
    control.put_json("_swarm/requests/r.json", {"id": "r", "verb": "submit", "payload": {
        "name": "wf", "stages": [{"agents": [{"prompt": "go", "schema": schema}]}]}})
    c.tick()
    box = sb.created[0]
    assert "conforms to this JSON schema" in box.started["prompt"]
    assert f"/shared/_swarm/outputs/{box.id}.json" in box.started["prompt"]
    # agent produces valid output, then finishes
    control.put_json(f"_swarm/outputs/{box.id}.json", {"summary": "done"})
    box._run.state = "done"
    c.tick()
    assert control.get_json(f"_swarm/agents/{box.id}.json")["schema_ok"] is True


def test_controller_unknown_verb_and_bad_plan_ack(tmp_path):
    c, control, _ = _controller(tmp_path)
    control.put_json("_swarm/requests/b.json", {"id": "b", "verb": "frobnicate", "payload": {}})
    control.put_json("_swarm/requests/p.json",
                     {"id": "p", "verb": "submit", "payload": {"stages": [{"agents": [{}]}]}})
    c.tick()
    assert control.get_json("_swarm/ack/b.json")["ok"] is False
    bad = control.get_json("_swarm/ack/p.json")
    assert bad["ok"] is False and any("prompt" in e for e in bad["errors"])


# --------------------------------------------------------------------------------------------- #
# durable traces (persist full parsed events to the volume so a done agent still renders)        #
# --------------------------------------------------------------------------------------------- #
class _Ev:
    def __init__(self, text):
        self._text = text

    def to_json(self):
        return {"kind": "text", "data": {"text": self._text}}


class _ParsingRun:
    """A run whose parse() turns each envelope into one text event."""

    def parse(self, env):
        return [_Ev(str(env))]


def test_controller_append_events_persists_full_parsed_trace(tmp_path):
    c, control, _ = _controller(tmp_path)
    run = _ParsingRun()
    c._append_events("agentX", run, ["a", "b"])
    lines = [ln for ln in (control.read_text("_swarm/traces/agentX.jsonl") or "").splitlines() if ln]
    assert len(lines) == 2
    assert json.loads(lines[0])["data"]["text"] == "a"
    # appends across calls (full trace), not truncated
    c._append_events("agentX", run, ["c"])
    lines = [ln for ln in control.read_text("_swarm/traces/agentX.jsonl").splitlines() if ln]
    assert [json.loads(ln)["data"]["text"] for ln in lines] == ["a", "b", "c"]


def test_controller_tails_orchestrator_trace(tmp_path):
    served = {"n": 0}

    class ORun(_ParsingRun):
        def read_trace(self, offset=0):
            served["n"] += 1
            return (["orch-env"], 1) if served["n"] == 1 else ([], offset)

    class OBox:
        def run(self, run_id):
            return ORun()

    class SB:
        def get(self, kind, sid):
            return OBox()

    control = LocalDirControlPlane(str(tmp_path))
    c = Controller(control, "sw", FakeEndpoint(), "aom-shared-sw", "claude",
                   orchestrator_id="orch-1", orchestrator_run_id="orch-1-run", sandboxes=SB())
    c._tail_orchestrator()  # resolves the orchestrator run via sandboxes.get(), tails + persists
    text = control.read_text("_swarm/traces/orch-1.jsonl")
    assert text and json.loads(text.splitlines()[0])["data"]["text"] == "orch-env"


# --------------------------------------------------------------------------------------------- #
# aom-swarm CLI                                                                                 #
# --------------------------------------------------------------------------------------------- #
def _run_cli(argv, swarm_dir, monkeypatch):
    monkeypatch.setenv("AOM_SWARM_DIR", str(swarm_dir))
    # The notes cursor is sandbox-local state; keep it inside the test's tree.
    monkeypatch.setenv("AOM_SWARM_NOTES_SEEN", str(Path(swarm_dir).parent / "_cli" / "notes.json"))
    monkeypatch.setattr(swarmctl.time, "sleep", lambda *_: None)
    out = io.StringIO()
    code = swarmctl.main(argv, out=out)
    return code, out.getvalue()


def test_cli_submit_rejects_bad_plan_before_sending(tmp_path, monkeypatch):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"stages": [{"agents": [{"goal": True}]}]}), encoding="utf-8")
    code, text = _run_cli(["submit", "--file", str(plan_file)], tmp_path / "_swarm", monkeypatch)
    assert code == 1 and "prompt" in text
    # nothing was sent to the controller
    assert not (tmp_path / "_swarm" / "requests").exists()


def test_cli_submit_writes_request_and_reads_ack(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-fixed")
    swarm_dir = tmp_path / "_swarm"
    # pre-seed the ack the CLI will poll for (stands in for the controller)
    LocalDirControlPlane(str(swarm_dir)).put_json("ack/req-fixed.json",
                                                  {"id": "req-fixed", "ok": True, "result": {"workflow": "wf"}})
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(
        {"stages": [{"name": "w1", "agents": [{"prompt": "x"}]}]}), encoding="utf-8")
    code, text = _run_cli(["submit", "--file", str(plan_file), "--name", "wf"], swarm_dir, monkeypatch)
    assert code == 0 and "ok" in text
    req = LocalDirControlPlane(str(swarm_dir)).get_json("requests/req-fixed.json")
    assert req["verb"] == "submit" and req["payload"]["name"] == "wf"


def test_cli_spawn_writes_request(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-sp")
    swarm_dir = tmp_path / "_swarm"
    code, _ = _run_cli(
        ["spawn", "--prompt", "hello", "--goal", "--timeout", "90", "--effort", "medium",
         "--ack-timeout", "0.1"],
        swarm_dir, monkeypatch,
    )
    assert code == 2  # no controller to ack; the request was still written
    req = LocalDirControlPlane(str(swarm_dir)).get_json("requests/req-sp.json")
    assert req["verb"] == "spawn"
    assert req["payload"]["agent"] == {"prompt": "hello", "goal": True, "timeout": 90.0, "effort": "medium"}


def test_cli_status_renders(tmp_path, monkeypatch):
    swarm_dir = tmp_path / "_swarm"
    LocalDirControlPlane(str(swarm_dir)).put_json("workflows/wf.json", {
        "name": "wf", "status": "running", "stages": [
            {"name": "w1", "status": "running", "max_concurrent": 2,
             "agents": [{"idx": 0, "id": "aom-1", "state": "running", "goal": True, "effort": "medium"}]}]})
    code, text = _run_cli(["status"], swarm_dir, monkeypatch)
    assert code == 0 and "workflow wf" in text and "w1" in text and "aom-1" in text


# --------------------------------------------------------------------------------------------- #
# DynamicSwarm.launch + reconnect                                                              #
# --------------------------------------------------------------------------------------------- #
def test_launch_spawns_orchestrator_with_injected_context(tmp_path, monkeypatch):
    created: list = []
    deployed: list = []
    scaled: list = []
    monkeypatch.setattr(model_ops, "ensure_deployed", lambda m, **k: (deployed.append(k), FakeEndpoint())[1])
    monkeypatch.setattr(model_ops, "deploy", lambda *a, **k: pytest.fail("launch must not redeploy"))
    monkeypatch.setattr(model_ops, "scale", lambda m, **k: scaled.append(k))
    monkeypatch.setattr(sandbox, "new_id", lambda prefix="": f"{prefix}test")

    def fake_create(name, **opts):
        box = FakeBox()
        box.create_opts = {"name": name, **opts}
        created.append(box)
        return box

    monkeypatch.setattr(sandbox, "create", fake_create)
    control = LocalDirControlPlane(str(tmp_path))

    ds = DynamicSwarm.launch("claude", "qwen3.8-27b", "do research", timeout=3600,
                             max_containers=1, control=control)

    assert ds.id == "swarm-test"
    assert ds.model == "qwen3.8-27b"
    assert ds.shared_volume == "aom-shared-swarm-test"
    assert ds.orchestrator_id == created[0].id
    started = created[0].started
    assert started["prompt"].startswith("# You are a dynamic-swarm orchestrator")
    assert "do research" in started["prompt"]
    assert started["goal"] is True
    assert started["meta"] == {"swarm_id": "swarm-test", "role": "orchestrator"}
    # the shared model is deployed (idempotent) but NOT scaled per swarm.
    assert deployed == [{"max_containers": 1}] and scaled == []
    meta = control.get_json("_swarm/meta.json")
    assert meta["orchestrator_id"] == created[0].id and meta["model"] == "qwen3.8-27b"
    # the launch record: the exact orchestrator prompt + what it takes to relaunch it
    assert control.read_text("_swarm/orchestrator_prompt.md") == started["prompt"]
    launch = control.get_json("_swarm/launch.json")
    assert launch["task"] == "do research" and launch["max_containers"] == 1
    assert launch["harness"] == "claude" and launch["model"] == "qwen3.8-27b"
    assert launch["proxied"] is False and launch["timeout"] == 3600


def test_reconnect_reads_meta(monkeypatch):
    class FakeVCP:
        def __init__(self, name):
            self.name = name

        def get_json(self, path):
            if path.endswith("meta.json"):
                return {"model": "qwen3.8-27b", "harness": "codex",
                        "orchestrator_id": "aom-9", "default_timeout": 1800}
            return None

    monkeypatch.setattr(dynamic, "VolumeControlPlane", lambda name: FakeVCP(name))
    ds = DynamicSwarm.reconnect("swarm-z")
    assert ds.id == "swarm-z" and ds.model == "qwen3.8-27b" and ds.harness == "codex"
    assert ds.orchestrator_id == "aom-9" and ds.default_timeout == 1800
    assert ds.shared_volume == "aom-shared-swarm-z"


def test_orchestrator_instructions_mention_cli_and_efforts():
    text = build_orchestrator_instructions(
        default_timeout=1800, allowed_efforts=["low", "medium", "xhigh"])
    for needle in ("aom-swarm submit", "aom-swarm wait", "max_concurrent", "/shared/_swarm",
                   "low, medium, xhigh", "1800"):
        assert needle in text


# --------------------------------------------------------------------------------------------- #
# UI wiring smoke                                                                               #
# --------------------------------------------------------------------------------------------- #
def test_index_html_wires_dynamic_mode():
    html = (EXAMPLES / "index.html").read_text(encoding="utf-8")
    for needle in ("/api/files", "/api/injected", "/api/agents", "/api/spawn",
                   "dynamic", "/api/presets", 'id="presetSel"', 'id="mode"',
                   'id="timeout"', 'id="effort"'):
        assert needle in html


def test_server_has_dynamic_routes():
    src = (EXAMPLES / "server.py").read_text(encoding="utf-8")
    for needle in ("/api/files", "/api/file", "/api/injected", "DynamicSwarm",
                   "_ensure_controller", "_dynamic_agents_payload"):
        assert needle in src


# --------------------------------------------------------------------------------------------- #
# feedback fixes: concurrent dispatch, settled agents, dead sandboxes, collapsed errors, schema  #
# paths, launch record                                                                          #
# --------------------------------------------------------------------------------------------- #
def _submit(control, plan, reqid="r"):
    control.put_json(f"_swarm/requests/{reqid}.json", {"id": reqid, "verb": "submit", "payload": plan})


def test_controller_dispatches_a_wave_concurrently(tmp_path):
    # A Modal create+start is ~10 s; one at a time a 100-agent wave took ~20 min to fill. Measure
    # how many creates overlap rather than wall time, so a loaded CI box can't flake it.
    state = {"live": 0, "peak": 0}
    lock = threading.Lock()

    class SlowSandboxes(FakeSandboxes):
        def create(self, name, **opts):
            with lock:
                state["live"] += 1
                state["peak"] = max(state["peak"], state["live"])
            time.sleep(0.2)
            with lock:
                state["live"] -= 1
            return super().create(name, **opts)

    c, control, sb = _controller(tmp_path, SlowSandboxes())
    _submit(control, {"name": "wf", "stages": [
        {"name": "w", "max_concurrent": 8, "agents": [{"prompt": f"p{i}"} for i in range(8)]}]})
    c.tick()
    assert len(sb.created) == 8
    assert state["peak"] > 1  # creates ran side by side, not one after another
    assert sorted(b.started["meta"]["idx"] for b in sb.created) == list(range(8))


def test_controller_stops_polling_settled_agents(tmp_path):
    calls = {"n": 0}

    class CountingRun(FakeRun):
        def status(self):
            calls["n"] += 1
            return super().status()

    class Boxes(FakeSandboxes):
        def create(self, name, **opts):
            box = super().create(name, **opts)
            box._run = CountingRun(run_id=f"{box.id}-run")
            return box

    c, control, sb = _controller(tmp_path, Boxes())
    _submit(control, {"name": "wf", "stages": [{"name": "w", "agents": [{"prompt": "a"}]}]})
    c.tick()
    sb.created[0]._run.state = "done"
    c.tick()  # sees "done"
    c.tick()  # one more pass flushes any trailing trace, then the agent is settled
    before = calls["n"]
    c.tick()
    c.tick()
    assert calls["n"] == before  # no more Modal round-trips for a finished agent
    assert control.get_json("_swarm/workflows/wf.json")["status"] == "done"


def test_controller_fails_an_agent_whose_sandbox_is_gone(tmp_path):
    # A sandbox killed from outside (timeout, idle timeout, OOM) can never report again; it must not
    # stay "running" forever and hang `aom-swarm wait` and its stage.
    c, control, sb = _controller(tmp_path)
    _submit(control, {"name": "wf", "stages": [{"name": "w", "agents": [{"prompt": "a"}]}]})
    c.tick()
    box = sb.created[0]

    def unreadable():
        raise RuntimeError("sandbox terminated")

    box._run.status = unreadable
    box.alive = lambda: True
    c.tick()  # unreadable but alive: a flaky read keeps the last state
    assert control.get_json(f"_swarm/agents/{box.id}.json")["state"] == "running"
    box.alive = lambda: False
    c.tick()
    agent = control.get_json(f"_swarm/agents/{box.id}.json")
    assert agent["state"] == "failed" and "sandbox exited" in agent["error"]
    assert control.get_json("_swarm/workflows/wf.json")["status"] == "done"


def test_controller_keeps_the_rest_of_a_wave_when_one_create_fails(tmp_path):
    class FlakySandboxes(FakeSandboxes):
        def __init__(self):
            super().__init__()
            self.failed = False
            self.lock = threading.Lock()

        def create(self, name, **opts):
            with self.lock:
                first, self.failed = not self.failed, True
            if first:
                raise RuntimeError("modal hiccup")
            return super().create(name, **opts)

    c, control, sb = _controller(tmp_path, FlakySandboxes())
    _submit(control, {"name": "wf", "stages": [
        {"name": "w", "max_concurrent": 4, "agents": [{"prompt": f"p{i}"} for i in range(4)]}]})
    with pytest.raises(RuntimeError, match="modal hiccup"):
        c.tick()  # surfaced (run_forever records it) ...
    assert len(sb.created) == 3  # ... but the three that did start are kept
    c.tick()
    assert sorted(b.started["meta"]["idx"] for b in sb.created) == list(range(4))  # retried


def test_parse_plan_collapses_an_error_repeated_across_a_wave():
    plan = {"stages": [{"agents": [{"prompt": "x", "schema": "/shared/schema.json"}] * 112}]}
    _, errors, _ = parse_plan(plan)
    assert len(errors) == 1
    assert errors[0].startswith("stages[0].agents[0-111] (112 agents): 'schema' must be a JSON object")
    assert dynamic._index_ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    _, one, _ = parse_plan({"stages": [{"agents": [{"prompt": "x", "schema": 1}]}]})
    assert one[0].startswith("stages[0].agents[0]: ")  # a single agent keeps the plain form


def test_cli_submit_inlines_a_shared_schema_file(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-s")
    swarm_dir = tmp_path / "_swarm"
    LocalDirControlPlane(str(swarm_dir)).put_json("ack/req-s.json", {"id": "req-s", "ok": True})
    schema = {"type": "object", "required": ["summary"]}
    (tmp_path / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"stages": [{"agents": [
        {"prompt": "a", "schema": "schema.json"}, {"prompt": "b", "schema": "schema.json"}]}]}),
        encoding="utf-8")
    code, _ = _run_cli(["submit", "--file", str(plan_file)], swarm_dir, monkeypatch)
    assert code == 0
    sent = LocalDirControlPlane(str(swarm_dir)).get_json("requests/req-s.json")["payload"]
    assert [a["schema"] for a in sent["stages"][0]["agents"]] == [schema, schema]


def test_cli_submit_reports_a_missing_schema_file_once(tmp_path, monkeypatch):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"stages": [{"agents": [
        {"prompt": "a", "schema": "nope.json"}] * 50}]}), encoding="utf-8")
    code, text = _run_cli(["submit", "--file", str(plan_file)], tmp_path / "_swarm", monkeypatch)
    assert code == 1 and "plan not submitted" in text
    assert text.count("schema file 'nope.json'") == 1  # one line, not one per agent
    assert text.count("(50 agents)") == 1  # and the per-agent lint error collapses to one line


def test_plan_agents_can_pick_their_own_harness(tmp_path):
    plan, errors, _ = parse_plan({"stages": [{"agents": [
        {"prompt": "a"}, {"prompt": "b", "harness": "codex"}]}]})
    assert not errors and [a.harness for a in plan.stages[0].agents] == [None, "codex"]
    _, bad, _ = parse_plan({"stages": [{"agents": [{"prompt": "a", "harness": "vim"}]}]})
    assert bad and "unknown harness 'vim'" in bad[0]

    c, control, sb = _controller(tmp_path)  # the swarm's harness is claude
    _submit(control, {"name": "wf", "stages": [{"name": "w", "agents": [
        {"prompt": "a"}, {"prompt": "b", "harness": "codex"}]}]})
    c.tick()
    by_idx = {b.started["meta"]["idx"]: b for b in sb.created}
    assert by_idx[0].started["harness"] == "claude" and by_idx[1].started["harness"] == "codex"
    assert control.get_json(f"_swarm/agents/{by_idx[1].id}.json")["harness"] == "codex"


def test_cli_wait_blocks_under_the_shell_timeout_and_reports_progress(tmp_path, monkeypatch):
    # A 7200 s default got the call killed by the harness's 120 s shell timeout, so orchestrators
    # polled `status` every second instead. One call must fit, then say where things stand.
    args = swarmctl.build_parser().parse_args(["wait"])
    assert args.timeout < 120
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    # The controller lists every started agent in its workflow file (wait reads only those).
    cp.put_json("workflows/wf.json", {"name": "wf", "status": "running", "stages": [
        {"name": "s", "agents": [{"id": "a1", "state": "done"}, {"id": "a2", "state": "running"}]}]})
    code, text = _run_cli(["wait", "--timeout", "0"], swarm_dir, monkeypatch)
    assert code == 2 and "1/2 agents finished" in text and "run `aom-swarm wait` again" in text


def test_plan_warns_when_agents_are_told_to_append_to_a_shared_file():
    # The task text beat the instructions: 62 of 100 agents' board lines were lost.
    _, errors, warnings = parse_plan({"stages": [{"agents": [
        {"prompt": "Append one line to /shared/board.md with your result"},
        {"prompt": "Append one line to /shared/board.md with your result"},
        {"prompt": "write /shared/board/<id>.md"}]}]})
    assert not errors
    assert len(warnings) == 1 and "(2 agents)" in warnings[0] and "last writer wins" in warnings[0]


def test_controller_delivers_acks_before_spawning_the_wave(tmp_path):
    # The ack used to reach the orchestrator only after the whole wave spawned (84.6 s of 90 s).
    order: list[str] = []

    class Boxes(FakeSandboxes):
        def create(self, name, **opts):
            order.append("spawn")
            return super().create(name, **opts)

    sb = Boxes()
    orch = sb.create("modal")
    order.clear()
    orch.reload_volumes = lambda: order.append("reload")
    control = LocalDirControlPlane(str(tmp_path))
    c = Controller(control, "sw", FakeEndpoint(), "aom-shared-sw", "claude",
                   orchestrator_id=orch.id, orchestrator_run_id="r", sandboxes=sb)
    _submit(control, {"name": "wf", "stages": [{"name": "w", "agents": [{"prompt": "a"}, {"prompt": "b"}]}]})
    c.tick()
    assert order[0] == "reload" and order.count("spawn") == 2


def test_cli_status_reports_an_unreadable_workflow_instead_of_dropping_it(tmp_path, monkeypatch):
    swarm_dir = tmp_path / "_swarm"
    (swarm_dir / "workflows").mkdir(parents=True)
    (swarm_dir / "workflows" / "wf.json").write_text("{ half-written", encoding="utf-8")
    code, text = _run_cli(["status", "--json"], swarm_dir, monkeypatch)
    assert '"unreadable"' in text


# --------------------------------------------------------------------------------------------- #
# hard stop (dashboard "stop swarm")                                                            #
# --------------------------------------------------------------------------------------------- #
def test_controller_halts_on_the_stop_marker_and_terminates_its_agents(tmp_path):
    c, control, sb = _controller(tmp_path)
    _submit(control, {"name": "wf", "stages": [
        {"name": "w1", "agents": [{"prompt": "a"}, {"prompt": "b"}]}, {"name": "w2", "agents": [{"prompt": "c"}]}]})
    c.tick()
    assert len(sb.created) == 2
    first, second = sorted(sb.created, key=lambda b: b.started["meta"]["idx"])
    first._run.state = "done"  # finished since the last tick: keeps "done", not "cancelled"

    control.put_json(f"_swarm/{dynamic.STOPPED_FILE}", {"ts": 1})
    c.tick()
    assert c.halted is True
    assert first.stopped and second.stopped  # every sandbox is terminated either way
    assert control.get_json(f"_swarm/agents/{first.id}.json")["state"] == "done"
    assert control.get_json(f"_swarm/agents/{second.id}.json")["state"] == "cancelled"
    assert control.get_json("_swarm/workflows/wf.json")["status"] == "cancelled"

    # nothing is ever scheduled again: not the next stage, not a new request
    second._run.state = "done"
    _submit(control, {"name": "wf2", "stages": [{"name": "s", "agents": [{"prompt": "x"}]}]}, reqid="r2")
    c.tick()
    assert len(sb.created) == 2 and not control.exists("_swarm/ack/r2.json")


def test_request_halt_stops_a_wave_mid_spawn_and_ends_run_forever(tmp_path):
    c, control, sb = _controller(tmp_path)
    _submit(control, {"name": "wf", "stages": [{"name": "w", "agents": [{"prompt": str(i)} for i in range(3)]}]})
    c.request_halt()
    with pytest.raises(dynamic.AomError):  # a spawn racing the stop creates no sandbox
        c._spawn_agent(dynamic._WorkflowRuntime(name="x", plan=parse_plan(
            {"stages": [{"name": "s", "agents": [{"prompt": "p"}]}]})[0]), 0, 0)
    t = threading.Thread(target=c.run_forever, kwargs={"interval": 0.01}, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive() and c.halted and sb.created == []


# --------------------------------------------------------------------------------------------- #
# feedback fixes: ack receipt, name dedupe, pending status, scoped stop, notes, settle, results  #
# --------------------------------------------------------------------------------------------- #
def test_ack_timeout_default_fits_the_shell_timeout():
    # Like `wait`: a call the harness's 120 s shell timeout kills never prints "do not resubmit".
    args = swarmctl.build_parser().parse_args(["submit", "--file", "x"])
    assert 60 <= args.ack_timeout < 120


def test_ack_timeout_points_at_the_request_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-fixed")
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"stages": [{"agents": [{"prompt": "x"}]}]}), encoding="utf-8")
    code, text = _run_cli(
        ["submit", "--file", str(plan_file), "--name", "wf", "--ack-timeout", "0.1"],
        tmp_path / "_swarm", monkeypatch,
    )
    assert code == 2
    assert "requests/req-fixed.json" in text and "very likely already accepted" in text
    assert "Do NOT resubmit" in text and "workflow wf" in text


def test_cli_submit_refuses_a_name_that_is_already_running(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-new")
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    cp.put_json("workflows/wf.json", {"name": "wf", "status": "running", "stages": []})
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"stages": [{"agents": [{"prompt": "x"}]}]}), encoding="utf-8")
    code, text = _run_cli(["submit", "--file", str(plan_file), "--name", "wf"], swarm_dir, monkeypatch)
    assert code == 1 and "already exists" in text and "--name wf-2" in text
    assert cp.listdir("requests") == []  # nothing was sent


def test_cli_submit_refuses_a_resubmit_of_a_pending_name_from_the_plan_file(tmp_path, monkeypatch):
    # "i1 was submitted five times": the name may come from the plan JSON, not only --name.
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    cp.put_json("requests/req-old.json", {"id": "req-old", "ts": time.time() - 30, "verb": "submit",
                                          "payload": {"name": "i1", "stages": []}})
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"name": "i1", "stages": [{"agents": [{"prompt": "x"}]}]}),
                         encoding="utf-8")
    code, text = _run_cli(["submit", "--file", str(plan_file)], swarm_dir, monkeypatch)
    assert code == 1 and "already pending (req-old" in text and "aom-swarm wait" in text
    assert cp.listdir("requests") == ["req-old.json"]
    # Once acked and handled it is no longer pending -- but the workflow now exists, still refused.
    cp.put_json("ack/req-old.json", {"id": "req-old", "ok": True})
    cp.put_json("workflows/i1.json", {"name": "i1", "status": "running", "stages": []})
    code, text = _run_cli(["submit", "--file", str(plan_file)], swarm_dir, monkeypatch)
    assert code == 1 and "already exists" in text


def test_cli_spawn_refuses_an_identical_pending_spawn_but_allows_a_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-n")
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    cp.put_json("requests/req-p.json", {"id": "req-p", "ts": time.time(), "verb": "spawn",
                                        "payload": {"workflow": "L1", "agent": {"prompt": "do L1"}}})
    code, text = _run_cli(["spawn", "--prompt", "do L1", "--timeout", "60", "--workflow", "L1"],
                          swarm_dir, monkeypatch)
    assert code == 1 and "identical spawn" in text
    cp.put_json("ack/req-n.json", {"id": "req-n", "ok": True, "result": {"workflow": "L1-2"}})
    code, text = _run_cli(["spawn", "--prompt", "redo L1 differently", "--timeout", "60",
                           "--workflow", "L1"], swarm_dir, monkeypatch)
    assert code == 0 and "L1-2" in text


def test_cli_spawn_attaches_the_default_schema_unless_told_not_to(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-d")
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    (tmp_path / "schema.json").write_text(json.dumps({"type": "object"}), encoding="utf-8")
    cp.put_json("ack/req-d.json", {"id": "req-d", "ok": True})
    code, text = _run_cli(["spawn", "--prompt", "x", "--timeout", "60", "--lane", "L1",
                           "--report-dir", "/shared/reports/L1"], swarm_dir, monkeypatch)
    sent = cp.get_json("requests/req-d.json")["payload"]["agent"]
    assert code == 0 and "campaign default" in text and sent["schema"] == {"type": "object"}
    assert sent["lane"] == "L1" and sent["report_dir"] == "/shared/reports/L1"
    (swarm_dir / "requests" / "req-d.json").unlink()
    code, text = _run_cli(["spawn", "--prompt", "setup", "--timeout", "60", "--no-schema"],
                          swarm_dir, monkeypatch)
    assert code == 0 and "schema" not in cp.get_json("requests/req-d.json")["payload"]["agent"]


def test_status_renders_pending_requests(tmp_path, monkeypatch):
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    cp.put_json("requests/req-p.json", {
        "id": "req-p", "ts": time.time(), "verb": "submit",
        "payload": {"name": "i1", "stages": [{"agents": [{"prompt": "a"}, {"prompt": "b"}]}]}})
    # A request the controller already claimed is not pending, even before its ack shows.
    cp.put_json("requests/req-h.json", {"id": "req-h", "ts": time.time(), "verb": "submit",
                                        "payload": {"name": "i0", "stages": []}})
    cp.put_json("handled/req-h.json", {"id": "req-h"})
    code, text = _run_cli(["status"], swarm_dir, monkeypatch)
    assert code == 0 and "PENDING submit i1" in text and "2 agent(s)" in text
    assert "Do not resubmit" in text and "i0" not in text


def test_status_flags_stuck_agents_on_the_controllers_clock_not_the_sandboxs(tmp_path, monkeypatch):
    # The sandbox clock can be hours off: age a controller-stamped field against now.json only.
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    controller_now = time.time() - 6 * 3600  # this "sandbox" runs six hours ahead
    cp.put_json("now.json", {"ts": controller_now})
    cp.put_json("workflows/wf.json", {"name": "wf", "status": "running", "stages": [
        {"name": "s", "planned": 3, "agents": [
            {"idx": 0, "id": "fresh", "state": "running", "updated_at": controller_now - 10},
            {"idx": 1, "id": "stale", "state": "running", "updated_at": controller_now - 900,
             "lane": "L2"}]}]})
    code, text = _run_cli(["status"], swarm_dir, monkeypatch)
    lines = {ln.split()[1]: ln for ln in text.splitlines() if ln.strip().startswith("[")}
    assert "STUCK" not in lines["fresh"] and "STUCK 15m" in lines["stale"]
    assert "lane=L2" in lines["stale"] and "2/3 agent(s) started" in text


def test_stop_prints_its_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(swarmctl, "_reqid", lambda: "req-s")
    swarm_dir = tmp_path / "_swarm"
    cp = LocalDirControlPlane(str(swarm_dir))
    cp.put_json("workflows/i0.json", {"name": "i0", "status": "running", "stages": [
        {"name": "s", "agents": [{"id": f"aom-{i}", "state": "done" if i < 2 else "running"}
                                 for i in range(3)]}]})
    cp.put_json("ack/req-s.json", {"id": "req-s", "ok": True})
    code, text = _run_cli(["stop", "i0"], swarm_dir, monkeypatch)
    assert code == 0 and "pausing workflow i0: 3 agent(s) started, 1 running" in text
    code, text = _run_cli(["stop", "aom-0"], swarm_dir, monkeypatch)
    assert code == 0 and "stopping agent aom-0 (workflow i0, done)" in text
    assert "already finished" in text


def test_wait_prints_new_supervisor_note_lines_once(tmp_path, monkeypatch):
    swarm_dir = tmp_path / "_swarm"
    note = tmp_path / "NOTE-supervisor.md"
    note.write_text("# notes\n\ni1 was submitted five times; stop resubmitting.\n", encoding="utf-8")
    code, text = _run_cli(["wait", "--timeout", "0"], swarm_dir, monkeypatch)
    assert code == 2 and "stop resubmitting" in text
    code2, text2 = _run_cli(["wait", "--timeout", "0"], swarm_dir, monkeypatch)
    assert code2 == 2 and "stop resubmitting" not in text2


def test_refresh_settles_a_run_whose_status_never_reads_again(tmp_path, monkeypatch):
    c, control, sb = _controller(tmp_path)
    _submit(control, {"name": "wf", "stages": [{"agents": [{"prompt": "x"}]}]})
    c.tick()
    ar = next(iter(c._workflows["wf"].runs[0].values()))

    def boom():
        raise RuntimeError("sandbox gone")

    ar.run.status = boom
    ar.last_status_ok = time.time() - 600
    # One failed read after a long gap (a slow 100-agent tick) is a blip, not a dead run: settling
    # it would start the next stage under a live lane.
    c.tick()
    assert ar.state == "running" and ar.refresh_errors == 1
    c.tick()
    c.tick()
    assert ar.state == "failed" and "unreadable" in ar.error


def test_parse_plan_accepts_lane_and_lints_a_mismatched_report_dir():
    plan, errors, warnings = parse_plan({"stages": [{"agents": [{
        "prompt": "x", "lane": "L1", "report_dir": "/shared/reports/i1/other"}]}]})
    assert errors == [] and plan.stages[0].agents[0].lane == "L1"
    assert any("does not appear in report_dir" in w for w in warnings)
    _, errors2, _ = parse_plan({"stages": [{"agents": [{
        "prompt": "x", "report_dir": "reports/i1/L1"}]}]})
    assert any("under /shared/" in e for e in errors2)
    # Any lane tree under /shared is fine; the control plane, traversal and multi-token lanes are not.
    _, ok, _ = parse_plan({"stages": [{"agents": [{"prompt": "x", "report_dir": "/shared/out/L1"}]}]})
    assert ok == []
    for bad in ({"report_dir": "/shared/_swarm/L1"}, {"report_dir": "/shared/r/../../etc"},
                {"lane": "two words"}):
        _, errs, _ = parse_plan({"stages": [{"agents": [{"prompt": "x", **bad}]}]})
        assert errs, bad


def test_cli_supervisor_appends_to_the_note(tmp_path, monkeypatch):
    cp = LocalDirControlPlane(str(tmp_path / "vol"))
    made = []
    monkeypatch.setattr(dynamic, "VolumeControlPlane",
                        lambda name, **k: made.append((name, k)) or cp)
    note_file = tmp_path / "n.md"
    note_file.write_text("stop resubmitting i1", encoding="utf-8")
    argv = ["supervisor", "--swarm", "sw", "--file", str(note_file)]
    # A mistyped swarm id must not create a volume and "deliver" the note into it.
    code, text = _run_cli(argv, tmp_path / "_swarm", monkeypatch)
    assert code == 1 and "no swarm 'sw'" in text and made[-1] == ("aom-shared-sw", {"create": False})
    cp.put_json("_swarm/meta.json", {"swarm_id": "sw"})
    code, text = _run_cli(argv, tmp_path / "_swarm", monkeypatch)
    assert code == 0 and "ok: appended" in text
    note = cp.read_text("NOTE-supervisor.md")
    assert "stop resubmitting i1" in note and "Supervisor notes" in note


def test_wait_labels_notes_and_handles_rewrites_and_large_notes(tmp_path, monkeypatch):
    swarm_dir = tmp_path / "_swarm"
    (tmp_path / "NOTE-L1.md").write_text("lane L1: skip the mirror\n", encoding="utf-8")
    _, text = _run_cli(["wait", "--timeout", "0"], swarm_dir, monkeypatch)
    assert "note NOTE-L1.md" in text and "outranks" not in text  # only the supervisor's does
    (tmp_path / "NOTE-L1.md").write_text("short\n", encoding="utf-8")  # rewritten, now smaller
    _, text = _run_cli(["wait", "--timeout", "0"], swarm_dir, monkeypatch)
    assert "(rewritten)" in text and "short" in text
    with open(tmp_path / "NOTE-L1.md", "a", encoding="utf-8") as f:
        f.write("é" * (swarmctl.NOTE_PRINT_CAP + 50))
    _, text = _run_cli(["wait", "--timeout", "0"], swarm_dir, monkeypatch)
    assert "more chars: read" in text and "short" not in text
    # The cursor is sandbox-local, never under the control dir.
    assert not (swarm_dir / "notes_seen.json").exists()


def test_controller_acks_a_request_that_lands_while_a_wave_is_being_created(tmp_path, monkeypatch):
    # Requests were read only between ticks, so one written during a 100-agent create waited out
    # the whole tick (90-105 s acks; every "timed out" retry double-ran the wave).
    gate = threading.Event()

    class SlowSandboxes(FakeSandboxes):
        def create(self, name, **opts):
            gate.wait(5)
            return super().create(name, **opts)

    c, control, sb = _controller(tmp_path, SlowSandboxes())
    monkeypatch.setattr(Controller, "INFLIGHT_POLL", 0.05)
    _submit(control, {"name": "big", "stages": [{"agents": [{"prompt": f"a{i}"} for i in range(4)]}]})
    t = threading.Thread(target=c.tick)
    t.start()
    deadline = time.time() + 5
    while not control.exists("_swarm/ack/r.json") and time.time() < deadline:
        time.sleep(0.01)
    _submit(control, {"name": "second", "stages": [{"agents": [{"prompt": "b"}]}]}, reqid="r2")
    while not control.exists("_swarm/ack/r2.json") and time.time() < deadline:
        time.sleep(0.01)
    acked_mid_wave = control.exists("_swarm/ack/r2.json") and t.is_alive()
    gate.set()
    t.join(5)
    assert acked_mid_wave and len(sb.created) == 4


def test_stop_acked_mid_wave_starts_no_more_of_that_workflow(tmp_path, monkeypatch):
    gate = threading.Event()

    class SlowSandboxes(FakeSandboxes):
        def create(self, name, **opts):
            gate.wait(5)
            return super().create(name, **opts)

    c, control, sb = _controller(tmp_path, SlowSandboxes())
    monkeypatch.setattr(Controller, "INFLIGHT_POLL", 0.05)
    monkeypatch.setattr(dynamic, "CONTROLLER_CONCURRENCY", 1)
    _submit(control, {"name": "big", "stages": [{"agents": [{"prompt": f"a{i}"} for i in range(5)]}]})
    t = threading.Thread(target=c.tick)
    t.start()
    while not control.exists("_swarm/ack/r.json"):
        time.sleep(0.01)
    control.put_json("_swarm/requests/r-stop.json", {"id": "r-stop", "verb": "stop",
                                                     "payload": {"target": "big"}})
    while not control.exists("_swarm/ack/r-stop.json"):
        time.sleep(0.01)
    gate.set()
    t.join(5)
    assert c._workflows["big"].paused and len(sb.created) <= 2


def test_stopping_a_finished_agent_keeps_its_outcome(tmp_path):
    c, control, sb = _controller(tmp_path)
    _submit(control, {"name": "wf", "stages": [{"agents": [{"prompt": "x"}]}]})
    c.tick()
    ar = next(iter(c._workflows["wf"].runs[0].values()))
    ar.run.state = "done"
    c.tick()
    control.put_json("_swarm/requests/r-stop.json", {"id": "r-stop", "verb": "stop",
                                                     "payload": {"target": ar.id}})
    c.tick()
    ack = control.get_json("_swarm/ack/r-stop.json")
    assert ack["ok"] and ack["result"]["status"] == "done" and ar.state == "done"
    assert control.get_json(f"_swarm/agents/{ar.id}.json")["state"] == "done"


def test_a_stopped_agent_stays_cancelled_even_if_an_in_flight_refresh_reads_running(tmp_path):
    c, control, sb = _controller(tmp_path)
    _submit(control, {"name": "wf", "stages": [{"agents": [{"prompt": "x"}]}]})
    c.tick()
    ar = next(iter(c._workflows["wf"].runs[0].values()))
    c._terminate(ar)
    c._refresh(ar)  # the fake still reports "running", as a read issued before the stop would
    assert ar.state == "cancelled"


def test_schema_fallback_reads_report_dir_result_and_promotes_it(tmp_path):
    c, control, sb = _controller(tmp_path)
    schema = {"type": "object", "required": ["summary"]}
    _submit(control, {"name": "wf", "stages": [{
        "agents": [{"prompt": "x", "schema": schema, "lane": "L1",
                    "report_dir": "/shared/reports/i1/L1"}]}]})
    c.tick()
    box = sb.created[0]
    assert "/shared/reports/i1/L1/result.json" in box.started["prompt"]
    ar = next(iter(c._workflows["wf"].runs[0].values()))
    control.put_json("reports/i1/L1/result.json", {"summary": "ok"})
    ar.run.state = "done"
    c.tick()
    assert ar.schema_ok is True
    assert control.get_json(f"_swarm/outputs/{ar.id}.json") == {"summary": "ok"}


def test_schema_is_rechecked_once_on_the_settling_refresh(tmp_path):
    # The agent's last write can trail its exit on the volume: a miss on the first terminal read
    # gets one more look before the run settles.
    c, control, sb = _controller(tmp_path)
    _submit(control, {"name": "wf", "stages": [{
        "agents": [{"prompt": "x", "schema": {"type": "object"}}]}]})
    c.tick()
    ar = next(iter(c._workflows["wf"].runs[0].values()))
    ar.run.state = "done"
    c.tick()
    assert ar.schema_ok is False and not ar.settled
    control.put_json(f"_swarm/outputs/{ar.id}.json", {"late": True})
    c.tick()
    assert ar.schema_ok is True and ar.settled


def test_read_shared_json_maps_sandbox_paths_onto_the_volume_root(tmp_path):
    # The controller's control plane is rooted at the volume (= /shared): "/shared/x" -> "x".
    cp = LocalDirControlPlane(str(tmp_path))
    cp.put_json("reports/L1/result.json", {"ok": 1})
    assert cp.read_shared_json("/shared/reports/L1/result.json") == {"ok": 1}
    assert cp.read_shared_json("/etc/passwd") is None
    assert cp.read_shared_json("/shared/../etc/passwd") is None


def test_controller_publishes_its_clock_at_most_every_reload_interval(tmp_path, monkeypatch):
    c, control, sb = _controller(tmp_path)
    writes = []
    real = control.put_json
    monkeypatch.setattr(control, "put_json",
                        lambda rel, obj: (writes.append(rel), real(rel, obj))[1])
    c.tick()
    c.tick()
    assert writes.count("_swarm/now.json") == 1
    assert abs(control.get_json("_swarm/now.json")["ts"] - time.time()) < 60


def test_a_failing_in_flight_callback_never_drops_created_agents(tmp_path, monkeypatch):
    # Dropping a batch's results after the creates succeeded leaks sandboxes nobody tracks: the
    # next tick creates the wave again (seen live: 120 orphans from a 12-agent wave).
    c, control, sb = _controller(tmp_path)
    monkeypatch.setattr(Controller, "INFLIGHT_POLL", 0.001)

    def boom():
        raise RuntimeError("in-flight callback failed")

    results = c._map(lambda i: (time.sleep(0.05), i)[1], [1, 2, 3], between=boom)
    assert [r for _, r, _ in results] == [1, 2, 3]
    _submit(control, {"name": "wf", "stages": [{"agents": [{"prompt": f"a{i}"} for i in range(3)]}]})
    monkeypatch.setattr(c, "_ingest_between", boom)
    c.tick()
    c.tick()
    assert len(sb.created) == 3 and len(c._workflows["wf"].runs[0]) == 3
