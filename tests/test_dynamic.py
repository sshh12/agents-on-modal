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
    cp.put_json("workflows/wf.json", {"name": "wf", "status": "running", "stages": []})
    cp.put_json("agents/a1.json", {"id": "a1", "workflow": "wf", "state": "done"})
    cp.put_json("agents/a2.json", {"id": "a2", "workflow": "wf", "state": "running"})
    code, text = _run_cli(["wait", "--timeout", "0"], swarm_dir, monkeypatch)
    assert code == 2 and "1/2 agents finished" in text and "run `aom-swarm wait` again" in text
