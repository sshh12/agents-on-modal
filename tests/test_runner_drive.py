"""Offline unit tests for ``agents_on_modal.runner``.

No real claude/codex, Docker or Modal. A fake harness drives tiny ``python -c`` subprocesses that
print scripted JSON lines (or sleep), and we assert the runner's trace, status transitions, goal
looping, deadline/cancel kills, and spawn-failure handling.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

import agents_on_modal.harness as harness_registry
from agents_on_modal import runner
from agents_on_modal.harness.base import Harness, HarnessSetup
from agents_on_modal.traces import Event


# --------------------------------------------------------------------------------------------------
# a fake harness
# --------------------------------------------------------------------------------------------------
def _script(lines: list[str], sleep: float = 0.0, exit_code: int = 0) -> str:
    """Build a python -c program that prints each JSON line, optionally sleeps, then exits."""

    body = ["import sys, time"]
    for line in lines:
        body.append(f"sys.stdout.write({line!r} + chr(10)); sys.stdout.flush()")
    if sleep:
        body.append(f"time.sleep({sleep})")
    body.append(f"sys.exit({exit_code})")
    return "\n".join(body)


class FakeHarness(Harness):
    name = "fake"
    protocol = "chat"
    executable = sys.executable

    def __init__(self, *, script: str, goal_active_calls: int = 0,
                 bad_cmd: bool = False, missing_session: bool = False) -> None:
        self._script = script
        self._goal_active_calls = goal_active_calls
        self._bad_cmd = bad_cmd
        self._missing_session = missing_session
        self.goal_calls = 0
        self.argv_calls = 0

    # Not exercised by the runner, but required by the ABC.
    def prepare(self, access, profile, *, home, workdir):  # pragma: no cover
        raise NotImplementedError

    def interactive_argv(self, setup):  # pragma: no cover
        return [self.executable]

    def headless_argv(self, setup, prompt, *, goal, resume):
        self.argv_calls += 1
        if self._bad_cmd:
            return ["this_binary_does_not_exist_aom_xyz"]
        return [sys.executable, "-c", self._script]

    def parse(self, envelope):
        data = envelope.get("data")
        if not isinstance(data, dict):
            return []
        kind = data.get("type")
        if kind == "session" and not self._missing_session:
            return [Event("session", {"session_id": data.get("session_id", "sess-1")})]
        if kind == "text":
            return [Event("text", {"text": data.get("text", "")})]
        if kind == "turn_end":
            return [Event("turn_end", {"result": data.get("result"), "is_error": data.get("is_error", False)})]
        return []

    def goal_state(self, setup, events, session_id):
        self.goal_calls += 1
        return "active" if self.goal_calls <= self._goal_active_calls else "complete"


def _write_spec(run_dir: Path, *, goal: bool = False, deadline: float | None = None,
                prompt: str = "do the thing") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    setup = {"harness": "fake", "home": str(run_dir / "harness"),
             "workdir": str(run_dir), "env": {}, "posix": True}
    (run_dir / "harness").mkdir(exist_ok=True)
    spec = {"run_id": run_dir.name, "harness": "fake", "model": "fake-model",
            "prompt": prompt, "goal": goal, "created_at": time.time(),
            "deadline": deadline, "setup": setup, "sandbox": {"kind": "local", "id": "x"}, "meta": {}}
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")


def _install(monkeypatch, harness: FakeHarness) -> None:
    monkeypatch.setattr(harness_registry, "get", lambda name: harness)


def _read_status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def _read_envelopes(run_dir: Path) -> list[dict]:
    text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _runner_events(run_dir: Path) -> list[dict]:
    return [e["data"] for e in _read_envelopes(run_dir) if e.get("src") == "runner"]


# --------------------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------------------
def test_non_goal_done_and_trace(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-1"
    script = _script(['{"type": "session", "session_id": "sess-A"}',
                      '{"type": "text", "text": "hello"}',
                      'plain non-json log line'])
    _install(monkeypatch, FakeHarness(script=script))
    _write_spec(run_dir, goal=False)

    rc = runner.main([str(run_dir)])
    assert rc == 0

    status = _read_status(run_dir)
    assert status["state"] == "done"
    assert status["exit_code"] == 0
    assert status["session_id"] == "sess-A"

    envs = _read_envelopes(run_dir)
    harness_envs = [e for e in envs if e.get("src") == "harness"]
    # JSON lines land in data, the non-JSON line lands in text.
    assert any(e.get("data", {}).get("type") == "text" for e in harness_envs)
    assert any("text" in e and "plain non-json" in e["text"] for e in harness_envs)
    events = [d.get("event") for d in _runner_events(run_dir)]
    assert events[0] == "start"
    assert "attempt" in events and "exit" in events and "done" in events


def test_status_progression_starting_running_done(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-2"
    # Sleep a little so the poller can observe the "running" state.
    script = _script(['{"type": "session", "session_id": "s"}'], sleep=0.4)
    _install(monkeypatch, FakeHarness(script=script))
    _write_spec(run_dir, goal=False)

    observed: list[str] = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                st = _read_status(run_dir)["state"]
                if not observed or observed[-1] != st:
                    observed.append(st)
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(0.01)

    watcher = threading.Thread(target=poll)
    watcher.start()
    rc = runner.main([str(run_dir)])
    stop.set()
    watcher.join()
    final = _read_status(run_dir)["state"]
    if not observed or observed[-1] != final:
        observed.append(final)

    assert rc == 0
    # "starting" is written synchronously in StatusWriter.__init__ before the loop, so it is only
    # on disk for microseconds; the poller reliably catches the running -> done transition.
    assert "running" in observed
    assert observed[-1] == "done"
    assert observed.index("running") < observed.index("done")


def test_status_writer_initial_state_is_starting(tmp_path):
    run_dir = tmp_path / "sw"
    run_dir.mkdir()
    runner.StatusWriter(run_dir)
    assert _read_status(run_dir)["state"] == "starting"


def test_goal_loops_until_complete(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-3"
    script = _script(['{"type": "session", "session_id": "sess-G"}',
                      '{"type": "text", "text": "progress"}'])
    # active for 2 calls, complete on the 3rd -> exactly 3 invocations.
    harness = FakeHarness(script=script, goal_active_calls=2)
    _install(monkeypatch, harness)
    _write_spec(run_dir, goal=True)

    rc = runner.main([str(run_dir)])
    assert rc == 0

    status = _read_status(run_dir)
    assert status["state"] == "done"
    assert status["goal_state"] == "complete"
    assert status["turns"] == 3
    assert harness.argv_calls == 3

    runner_evs = _runner_events(run_dir)
    # First invocation is "attempt", the next two are "resume".
    assert sum(1 for d in runner_evs if d.get("event") == "attempt") == 1
    assert sum(1 for d in runner_evs if d.get("event") == "resume") == 2
    goal_checks = [d for d in runner_evs if d.get("event") == "goal_check"]
    assert [d["goal_state"] for d in goal_checks] == ["active", "active", "complete"]


def test_goal_no_progress_stops(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-np"
    # Emit the session once then nothing new; goal never completes.
    script = _script(['{"type": "session", "session_id": "sess-N"}'])
    harness = FakeHarness(script=script, goal_active_calls=1000)
    _install(monkeypatch, harness)
    _write_spec(run_dir, goal=True)

    rc = runner.main([str(run_dir)])
    assert rc == 0
    status = _read_status(run_dir)
    assert status["state"] == "done"
    # Every invocation emits only a (repeating) session line, so none makes progress.
    assert harness.argv_calls == runner.MAX_NO_PROGRESS
    assert any(d.get("reason") == "no-progress" for d in _runner_events(run_dir))


def test_deadline_times_out_and_kills_child(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-4"
    # Child would run for 30s; the deadline must kill it fast.
    script = _script(['{"type": "session", "session_id": "s"}'], sleep=30.0)
    _install(monkeypatch, FakeHarness(script=script))
    _write_spec(run_dir, goal=False, deadline=time.time() + 0.4)

    start = time.time()
    rc = runner.main([str(run_dir)])
    elapsed = time.time() - start

    assert rc == 0
    assert elapsed < 15  # proves the child was killed, not waited out
    assert _read_status(run_dir)["state"] == "timeout"
    assert any(d.get("event") == "timeout" for d in _runner_events(run_dir))


def test_cancel_file_cancels_and_kills_child(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-5"
    script = _script(['{"type": "session", "session_id": "s"}'], sleep=30.0)
    _install(monkeypatch, FakeHarness(script=script))
    _write_spec(run_dir, goal=False)

    def cancel_soon():
        time.sleep(0.4)
        (run_dir / "cancel").write_text("cancel\n", encoding="utf-8")

    canceller = threading.Thread(target=cancel_soon)
    canceller.start()
    start = time.time()
    rc = runner.main([str(run_dir)])
    elapsed = time.time() - start
    canceller.join()

    assert rc == 0
    assert elapsed < 15
    assert _read_status(run_dir)["state"] == "cancelled"


def test_cancel_file_present_before_start(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-6"
    script = _script(['{"type": "session", "session_id": "s"}'])
    harness = FakeHarness(script=script)
    _install(monkeypatch, harness)
    _write_spec(run_dir, goal=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cancel").write_text("cancel\n", encoding="utf-8")

    rc = runner.main([str(run_dir)])
    assert rc == 0
    assert _read_status(run_dir)["state"] == "cancelled"
    # The cancel was seen before any invocation.
    assert harness.argv_calls == 0


def test_spawn_failure_is_failed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-7"
    _install(monkeypatch, FakeHarness(script="", bad_cmd=True))
    _write_spec(run_dir, goal=False)

    rc = runner.main([str(run_dir)])
    assert rc == 1
    status = _read_status(run_dir)
    assert status["state"] == "failed"
    assert "could not start harness" in status.get("error", "")


def test_nonzero_exit_without_result_is_failed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-8"
    script = _script(['{"type": "session", "session_id": "s"}'], exit_code=3)
    _install(monkeypatch, FakeHarness(script=script))
    _write_spec(run_dir, goal=False)

    rc = runner.main([str(run_dir)])
    assert rc == 1
    status = _read_status(run_dir)
    assert status["state"] == "failed"
    assert status["exit_code"] == 3


def test_nonzero_exit_with_result_is_done(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-9"
    script = _script(['{"type": "session", "session_id": "s"}',
                      '{"type": "turn_end", "result": "ok", "is_error": false}'],
                     exit_code=2)
    _install(monkeypatch, FakeHarness(script=script))
    _write_spec(run_dir, goal=False)

    rc = runner.main([str(run_dir)])
    assert rc == 0
    assert _read_status(run_dir)["state"] == "done"


def test_bad_harness_line_does_not_crash(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-10"
    # A valid-JSON line the parser does not understand, plus garbage -> both tolerated.
    script = _script(['{"weird": true}', 'not json at all {', '{"type": "session", "session_id": "s"}'])
    _install(monkeypatch, FakeHarness(script=script))
    _write_spec(run_dir, goal=False)

    rc = runner.main([str(run_dir)])
    assert rc == 0
    assert _read_status(run_dir)["state"] == "done"


def test_missing_run_dir_argument(tmp_path, monkeypatch):
    assert runner.main([]) == 2


def test_bad_spec_records_failed(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-11"
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text("{not valid json", encoding="utf-8")
    rc = runner.main([str(run_dir)])
    assert rc == 1
    assert _read_status(run_dir)["state"] == "failed"
