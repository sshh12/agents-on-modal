"""Offline tests for dynamic swarms: plan model, pure scheduler, control plane, the controller
(pool + waves + stop/resume + schema), the ``aom-swarm`` CLI, and ``DynamicSwarm.launch``.

No Modal. ``sandbox``/``models`` seams are faked or monkeypatched exactly like ``test_swarm.py``.
"""

from __future__ import annotations

import io
import json
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
    _n = 0

    def __init__(self):
        FakeBox._n += 1
        self.id = f"aom-{FakeBox._n:04d}"
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
    a0 = sb.created[0].started
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
    monkeypatch.setattr(model_ops, "deploy", lambda m: (deployed.append(m), FakeEndpoint())[1])
    monkeypatch.setattr(model_ops, "scale", lambda m, **k: scaled.append(k))
    monkeypatch.setattr(sandbox, "new_id", lambda prefix="": f"{prefix}test")

    def fake_create(name, **opts):
        box = FakeBox()
        box.create_opts = {"name": name, **opts}
        created.append(box)
        return box

    monkeypatch.setattr(sandbox, "create", fake_create)
    control = LocalDirControlPlane(str(tmp_path))

    ds = DynamicSwarm.launch("claude", "qwen3.8-27b", "do research", timeout=3600, control=control)

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
    assert deployed and scaled == []
    meta = control.get_json("_swarm/meta.json")
    assert meta["orchestrator_id"] == created[0].id and meta["model"] == "qwen3.8-27b"


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
        default_timeout=1800, model_name="qwen3.8-27b", allowed_efforts=["low", "medium", "xhigh"])
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
