"""Regression tests for the platform issues in ``C:\\tmp\\harness-issues.md`` (S1-S10).

Offline only -- no Modal, no real harness. Each test names the issue it guards. The live Modal e2e
that exercises S1/S2/S5 end to end is run separately (see the issue doc's Fix notes); these pin the
logic so it cannot silently regress.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

import agents_on_modal
import agents_on_modal.dynamic as dynamic
import agents_on_modal.harness as harness_registry
from agents_on_modal import runner
from agents_on_modal.dynamic import Controller
from agents_on_modal.gateway import ir
from agents_on_modal.harness.base import Harness
from agents_on_modal.traces import Event


# --------------------------------------------------------------------------------------------- #
# S1 -- gateway strips image blocks for a text-only model                                       #
# --------------------------------------------------------------------------------------------- #
def test_s1_drop_images_replaces_blocks_and_is_idempotent():
    messages = [
        {"role": "user", "content": "plain text"},
        {"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]},
        {"role": "tool", "tool_call_id": "t1", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
        ]},
        {"role": "assistant", "content": "ok"},
    ]
    changed = ir.drop_images_for_text_model(messages)
    assert changed == 2
    # no image block survives anywhere
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            assert not any(isinstance(b, dict) and b.get("type") in ("image", "image_url") for b in c)
    assert messages[1]["content"] == "look at this\n" + ir.IMAGE_OMITTED
    assert messages[2]["content"] == ir.IMAGE_OMITTED        # image-only content -> placeholder
    assert messages[0]["content"] == "plain text"            # text untouched
    assert messages[3]["content"] == "ok"
    assert ir.drop_images_for_text_model(messages) == 0       # idempotent


def test_s1_anthropic_request_pipeline_strips_images():
    from agents_on_modal.gateway import anthropic

    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
    ]}]}
    parsed = anthropic.parse_request(body)

    def has_image(msgs):
        return any(isinstance(m.get("content"), list)
                   and any(isinstance(b, dict) and b.get("type") in ("image", "image_url")
                           for b in m["content"])
                   for m in msgs)

    assert has_image(parsed.messages)                 # the rendered request carries an image block
    ir.drop_images_for_text_model(parsed.messages)
    assert not has_image(parsed.messages)             # ... and it is gone after the S1 strip


# --------------------------------------------------------------------------------------------- #
# S2 -- a failed compaction is retried, not counted as a no-progress stall                      #
# --------------------------------------------------------------------------------------------- #
def test_s2_detects_compaction_signatures():
    assert runner._looks_like_compaction_failure(
        [Event("turn_end", {"result": "API Error: 400 prompt is too long", "is_error": True})])
    assert runner._looks_like_compaction_failure(
        [Event("error", {"message": "validate_thinking_parts_role: content should be a valid string"})])
    assert not runner._looks_like_compaction_failure(
        [Event("text", {"text": "the plan is long but fine"}), Event("tool_call", {"input": {}})])


def _prog(lines: list[str], exit_code: int) -> str:
    body = ["import sys"]
    for ln in lines:
        body.append(f"sys.stdout.write({ln!r} + chr(10)); sys.stdout.flush()")
    body.append(f"sys.exit({exit_code})")
    return "\n".join(body)


class _CompactionHarness(Harness):
    """Fails the first ``fail_attempts`` attempts with a compaction 400, then succeeds."""

    name = "fake"
    protocol = "chat"
    executable = sys.executable

    def __init__(self, fail_attempts: int) -> None:
        self.fail_attempts = fail_attempts
        self.argv_calls = 0
        self.goal_calls = 0

    def prepare(self, *a, **k):  # pragma: no cover - not exercised
        raise NotImplementedError

    def interactive_argv(self, setup):  # pragma: no cover
        return [self.executable]

    def headless_argv(self, setup, prompt, *, goal, resume):
        self.argv_calls += 1
        if self.argv_calls <= self.fail_attempts:
            lines = ['{"type": "session", "session_id": "s"}',
                     '{"type": "turn_end", "result": "API Error: 400 prompt is too long", "is_error": true}']
            return [sys.executable, "-c", _prog(lines, 1)]
        lines = ['{"type": "session", "session_id": "s"}',
                 '{"type": "turn_end", "result": "ok", "is_error": false}']
        return [sys.executable, "-c", _prog(lines, 0)]

    def parse(self, envelope):
        data = envelope.get("data")
        if not isinstance(data, dict):
            return []
        if data.get("type") == "session":
            return [Event("session", {"session_id": data.get("session_id", "s")})]
        if data.get("type") == "turn_end":
            return [Event("turn_end", {"result": data.get("result"), "is_error": data.get("is_error", False)})]
        return []

    def goal_state(self, setup, events, session_id):
        self.goal_calls += 1
        return "active" if self.goal_calls <= self.fail_attempts else "complete"


def _write_goal_spec(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "harness").mkdir(exist_ok=True)
    setup = {"harness": "fake", "home": str(run_dir / "harness"), "workdir": str(run_dir),
             "env": {}, "posix": True}
    spec = {"run_id": run_dir.name, "harness": "fake", "model": "m", "prompt": "go", "goal": True,
            "created_at": time.time(), "deadline": None, "setup": setup,
            "sandbox": {"kind": "local", "id": "x"}, "meta": {}}
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")


def _runner_events(run_dir: Path) -> list[dict]:
    text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    return [json.loads(l)["data"] for l in text.splitlines()
            if l.strip() and json.loads(l).get("src") == "runner"]


def test_s2_compaction_failures_retried_not_treated_as_stall(tmp_path, monkeypatch):
    # 4 consecutive compaction failures (> MAX_NO_PROGRESS=3) then success. Without the fix the
    # no-progress guard ends the run at attempt 3; with it, the run survives to completion.
    harness = _CompactionHarness(fail_attempts=4)
    monkeypatch.setattr(harness_registry, "get", lambda name: harness)
    run_dir = tmp_path / "run-s2"
    _write_goal_spec(run_dir)

    rc = runner.main([str(run_dir)])
    assert rc == 0

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done"
    assert status["goal_state"] == "complete"
    assert harness.argv_calls == 5  # 4 failures + 1 success -- never gave up early

    events = _runner_events(run_dir)
    names = [e.get("event") for e in events]
    assert names.count("compaction_retry") == 3            # bounded by MAX_COMPACTION_RETRIES
    assert "compaction_retry_exhausted" in names           # the 4th failure spent the budget
    assert "no-progress" not in [e.get("reason") for e in events]  # never ended as a stall


# --------------------------------------------------------------------------------------------- #
# shared fakes for the controller tests                                                         #
# --------------------------------------------------------------------------------------------- #
class _FakeModel:
    name = "m"


class _FakeEndpoint:
    def __init__(self):
        self.model = _FakeModel()
        self.url = "https://x.modal.direct"
        self.token = "wk-a.ws-b"


class _FakeRun:
    def __init__(self, box_id):
        self.run_id = f"{box_id}-run"
        self.state = "running"

    def status(self):
        return {"state": self.state}

    def read_trace(self, offset=0):
        return [], offset


class _FakeBox:
    _n = 0

    def __init__(self):
        _FakeBox._n += 1
        self.id = f"aom-{_FakeBox._n:04d}"
        self._run = _FakeRun(self.id)

    def start(self, harness, endpoint, prompt, **kw):
        self._run.started = {"harness": harness, "prompt": prompt, **kw}
        return self._run

    def run(self, run_id):
        return self._run

    def write_file(self, *a, **k):
        pass

    def exec(self, *a, **k):
        return ""

    def stop(self):
        pass


class _FakeSB:
    def __init__(self):
        self.created = []

    def create(self, name, **opts):
        box = _FakeBox()
        self.created.append(box)
        return box

    def get(self, name, sid):
        for b in self.created:
            if b.id == sid:
                return b
        raise KeyError(sid)


# --------------------------------------------------------------------------------------------- #
# S5 -- a request is executed at most once (dedupe at dispatch), even across a restart           #
# --------------------------------------------------------------------------------------------- #
def test_s5_claims_request_at_dispatch_and_no_duplicate_on_restart(tmp_path):
    control = dynamic.LocalDirControlPlane(str(tmp_path))
    plan = {"stages": [{"name": "w", "agents": [{"prompt": "x"}]}]}
    control.put_json("_swarm/requests/r.json", {"id": "r", "verb": "submit", "payload": plan})

    c1 = Controller(control, "sw", _FakeEndpoint(), "aom-shared-sw", "claude", sandboxes=_FakeSB())
    c1.tick()
    assert control.get_json("_swarm/ack/r.json")["ok"] is True
    assert control.get_json("_swarm/handled/r.json")["verb"] == "submit"  # claimed at dispatch

    # The workflow is named after the request id ('r') since the plan carries no top-level name.
    assert sorted(n for n in control.listdir("_swarm/workflows")) == ["r.json"]

    # Simulate a controller restart: a brand-new Controller (empty in-memory state), same volume.
    c2 = Controller(control, "sw", _FakeEndpoint(), "aom-shared-sw", "claude", sandboxes=_FakeSB())
    c2.tick()
    assert sorted(n for n in control.listdir("_swarm/workflows")) == ["r.json"]  # no 'r-2'
    assert "r" not in c2._workflows and "r-2" not in c2._workflows


def test_s5_handled_marker_blocks_reexecution_even_without_ack(tmp_path):
    # The exact S5 condition: a prior pass claimed+ran the request but crashed before writing the
    # ack. With only the old ack-existence check it would re-run as a duplicate; the handled marker
    # must stop that.
    control = dynamic.LocalDirControlPlane(str(tmp_path))
    plan = {"stages": [{"name": "w", "agents": [{"prompt": "x"}]}]}
    control.put_json("_swarm/requests/r.json", {"id": "r", "verb": "submit", "payload": plan})
    control.put_json("_swarm/handled/r.json", {"id": "r", "verb": "submit", "ts": 0})  # no ack

    c = Controller(control, "sw", _FakeEndpoint(), "aom-shared-sw", "claude", sandboxes=_FakeSB())
    c.tick()
    assert [n for n in control.listdir("_swarm/workflows")] == []  # not re-executed
    assert "w" not in c._workflows


# --------------------------------------------------------------------------------------------- #
# S4 -- CLI ack-timeout default raised, env-overridable                                          #
# --------------------------------------------------------------------------------------------- #
def test_s4_ack_timeout_default_and_env(monkeypatch):
    import agents_on_modal.swarmctl as swarmctl

    monkeypatch.delenv("AOM_SWARM_ACK_TIMEOUT", raising=False)
    args = swarmctl.build_parser().parse_args(["spawn", "--prompt", "x", "--timeout", "1"])
    assert args.ack_timeout == 900.0  # was 120 -- past the ~12 min observed latency

    monkeypatch.setenv("AOM_SWARM_ACK_TIMEOUT", "77")
    args = swarmctl.build_parser().parse_args(["submit", "--file", "-"])
    assert args.ack_timeout == 77.0


# --------------------------------------------------------------------------------------------- #
# S9 -- control-plane re-exports; orchestrator session id published to meta                      #
# --------------------------------------------------------------------------------------------- #
def test_s9_control_plane_reexported_at_top_level():
    assert agents_on_modal.VolumeControlPlane is dynamic.VolumeControlPlane
    assert agents_on_modal.LocalDirControlPlane is dynamic.LocalDirControlPlane
    assert agents_on_modal.ControlPlane is dynamic.ControlPlane


def test_s9_orchestrator_session_published_to_meta(tmp_path):
    control = dynamic.LocalDirControlPlane(str(tmp_path))

    class _OrchRun:
        def __init__(self):
            self._served = False

        def read_trace(self, offset):
            if self._served:
                return [], offset
            self._served = True
            return [{"data": {"type": "session", "session_id": "sess-xyz"}}], offset + 1

        def status(self):
            return {"state": "running"}

        def parse(self, env):
            d = env.get("data") or {}
            if d.get("type") == "session":
                return [Event("session", {"session_id": d.get("session_id")})]
            return []

    class _OrchBox:
        id = "orch-1"

        def run(self, rid):
            return _OrchRun()

    class _SB:
        def get(self, name, sid):
            return _OrchBox()

    c = Controller(control, "sw", _FakeEndpoint(), "aom-shared-sw", "claude",
                   orchestrator_id="orch-1", orchestrator_run_id="orch-1-run", sandboxes=_SB())
    c.tick()
    meta = control.get_json("_swarm/meta.json")
    assert meta.get("orchestrator_session") == "sess-xyz"


# --------------------------------------------------------------------------------------------- #
# S3 -- host-usable orchestrator resume: fresh relaunch carrying a state snapshot                #
# --------------------------------------------------------------------------------------------- #
def test_s3_resume_orchestrator_relaunches_with_state(tmp_path, monkeypatch):
    control = dynamic.LocalDirControlPlane(str(tmp_path))
    control.put_json("_swarm/meta.json", {
        "swarm_id": "sw", "model": "m", "harness": "claude",
        "orchestrator_id": "old-orch", "orchestrator_run_id": "old-run", "default_timeout": 3600})
    control.put_json("_swarm/workflows/w.json", {
        "name": "w", "status": "running",
        "stages": [{"name": "s1", "status": "running",
                    "agents": [{"idx": 0, "state": "done"}, {"idx": 1, "state": "running"}]}]})

    monkeypatch.setattr(dynamic, "VolumeControlPlane", lambda name: control)
    fake_model = type("M", (), {"name": "m", "reasoning": type("R", (), {"efforts": {}})()})()
    monkeypatch.setattr(agents_on_modal.models, "get", lambda n: fake_model)
    monkeypatch.setattr(agents_on_modal.models, "endpoint", lambda m: _FakeEndpoint())

    sb = _FakeSB()
    result = dynamic.resume_orchestrator("sw", sandboxes=sb)

    assert result["resumed_from"] == "old-orch"
    assert result["orchestrator_id"] == sb.created[0].id
    meta = control.get_json("_swarm/meta.json")
    assert meta["orchestrator_id"] == sb.created[0].id
    assert meta["orchestrator_run_id"].endswith("-run")
    assert meta["resumed_from"] == "old-orch"

    started = sb.created[0]._run.started
    assert started["goal"] is True                              # orchestrator must persist
    assert "Resuming an in-flight mission" in started["prompt"]  # resume framing
    assert "workflow `w`" in started["prompt"]                   # state snapshot included
    assert "[1/2 agents done]" in started["prompt"]


# --------------------------------------------------------------------------------------------- #
# S7 -- dashboard /api/file pages a large file instead of cutting it mid-content                 #
# --------------------------------------------------------------------------------------------- #
def test_s7_file_payload_pages_large_file(monkeypatch):
    swarm_dir = str(Path(__file__).resolve().parents[1] / "examples" / "swarm")
    if swarm_dir not in sys.path:
        sys.path.insert(0, swarm_dir)
    import server  # examples/swarm/server.py

    blob = b"A" * 500_000
    monkeypatch.setattr(server.Swarm, "read_shared", lambda self, path: blob, raising=True)

    first = server._file_payload("sw", "trace.jsonl", 0)
    assert first["size"] == 500_000
    assert first["offset"] == 0
    assert len(first["text"]) == server.MAX_FILE_BYTES
    assert first["eof"] is False
    assert first["truncated"] is True

    # paging through the windows reconstructs the whole file exactly
    chunks, off = [], 0
    while True:
        page = server._file_payload("sw", "trace.jsonl", off)
        chunks.append(page["text"])
        off = page["next_offset"]
        if page["eof"]:
            break
    assert "".join(chunks) == "A" * 500_000
    assert off == 500_000


# --------------------------------------------------------------------------------------------- #
# S10 -- UTF-8 console setup never crashes and exports PYTHONIOENCODING                           #
# --------------------------------------------------------------------------------------------- #
def test_s10_ensure_utf8_io_sets_env_and_reconfigures(monkeypatch):
    from agents_on_modal import config

    calls = []

    class _Stream:
        encoding = "cp1252"

        def reconfigure(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(sys, "stdout", _Stream())
    monkeypatch.setattr(sys, "stderr", _Stream())
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    config.ensure_utf8_io()
    assert os.environ.get("PYTHONIOENCODING") == "utf-8"
    assert calls and all(c.get("encoding") == "utf-8" and c.get("errors") == "replace" for c in calls)

    calls.clear()
    # already UTF-8 on both streams -> no reconfigure, no raise
    utf8 = type("S", (), {"encoding": "utf-8", "reconfigure": lambda self, **k: calls.append(k)})
    monkeypatch.setattr(sys, "stdout", utf8())
    monkeypatch.setattr(sys, "stderr", utf8())
    config.ensure_utf8_io()
    assert calls == []
