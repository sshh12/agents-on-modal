"""In-sandbox run driver: ``python -m agents_on_modal.runner <run_dir>``.

This process runs *inside* every sandbox. It reads ``<run_dir>/spec.json``, reconstructs the harness
and its :class:`~agents_on_modal.harness.base.HarnessSetup`, launches the harness headless, tees its
JSON stream into ``<run_dir>/trace.jsonl`` (one envelope per line, see
:mod:`agents_on_modal.traces`), drives goal-mode continuation, enforces the deadline, honours a
``<run_dir>/cancel`` file, and rewrites ``<run_dir>/status.json`` atomically on every transition.

Everything a reader needs is on disk, so :class:`~agents_on_modal.sandbox.base.Run` can reattach from
any process. The runner never calls the network itself and never crashes on a bad harness line.

Loop control:

* **Non-goal run** -- one invocation; ``done`` when it exits cleanly, ``failed`` on a nonzero exit
  that produced no result.
* **Goal run** -- keep resuming the session (``harness.continuation_prompt(goal=True)``,
  ``resume=session_id``) until the harness reports the goal ``complete``/``blocked``, the deadline
  passes, a cancel file appears, or there are too many consecutive no-progress iterations. Claude's
  ``headless_argv(goal=True)`` self-loops via ``/goal`` so one invocation often suffices; Codex's
  ``exec`` does not auto-continue, so the runner loops ``exec resume``.

The runner is cross-platform: it ships into Linux containers but the ``local`` sandbox runs it on the
host, which may be Windows. It uses ``start_new_session`` on POSIX and ``CREATE_NEW_PROCESS_GROUP`` on
win32, and a kill helper that works on both. No ``os.fork`` / ``os.killpg`` on Windows.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import harness as harness_registry
from . import traces
from .harness.base import GoalState, Harness, HarnessSetup

#: States that mean the run is over (mirrors ``sandbox.base.TERMINAL_STATES``).
TERMINAL_STATES = ("done", "timeout", "failed", "cancelled")

#: How often to poll for deadline / cancel while a child runs (seconds).
POLL_INTERVAL = float(os.environ.get("AOM_RUNNER_POLL", "0.1"))

#: Consecutive goal iterations with no new distinct action before we give up.
MAX_NO_PROGRESS = 3


def _distinct_actions(events: list) -> int:
    """How many *distinct* tool calls the agent has made so far.

    Goal progress is measured by distinct actions, not by output volume or raw tool-call count: a
    finished agent that keeps narrating, or keeps re-running the same verification command on each
    resume, makes no new distinct action and so terminates. An agent genuinely still working issues
    new, different tool calls, so an impossible goal keeps progressing until the deadline.
    """

    seen = set()
    for event in events:
        if event.kind == "tool_call":
            seen.add(json.dumps(event.data.get("input"), sort_keys=True, default=str))
    return len(seen)

#: Child subprocess currently running, so a signal handler can kill its tree.
_ACTIVE: dict[str, "subprocess.Popen[str] | None"] = {"proc": None}


class _SpawnError(Exception):
    """The harness subprocess could not be started at all."""


# --------------------------------------------------------------------------------------------------
# status.json
# --------------------------------------------------------------------------------------------------
class StatusWriter:
    """Holds the run's status and rewrites ``status.json`` atomically on each change."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "status.json"
        now = time.time()
        self._state: dict[str, Any] = {
            "state": "starting",
            "started_at": now,
            "updated_at": now,
            "session_id": None,
            "turns": 0,
            "goal_state": None,
            "exit_code": None,
        }
        self._write()

    def set(self, **fields: Any) -> None:
        """Merge ``fields`` in and persist. ``error=None`` removes any prior error."""

        for key, value in fields.items():
            if key == "error" and value is None:
                self._state.pop("error", None)
            else:
                self._state[key] = value
        self._state["updated_at"] = time.time()
        self._write()

    @property
    def state(self) -> str:
        return str(self._state["state"])

    def _write(self) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)


# --------------------------------------------------------------------------------------------------
# trace.jsonl
# --------------------------------------------------------------------------------------------------
class TraceSink:
    """Append-only, thread-safe writer for trace envelopes."""

    def __init__(self, run_dir: Path) -> None:
        # newline="\n" keeps LF everywhere (containers and Windows host alike).
        self._f = open(run_dir / "trace.jsonl", "a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def emit_harness(self, line: str) -> dict[str, Any]:
        """Wrap one raw harness stdout/stderr line and append it."""

        env = traces.envelope("harness", line)
        self._write(env)
        return env

    def emit_runner(self, data: dict[str, Any]) -> dict[str, Any]:
        """Append a runner-lifecycle envelope."""

        env = traces.envelope("runner", data)
        self._write(env)
        return env

    def _write(self, env: dict[str, Any]) -> None:
        payload = json.dumps(env, ensure_ascii=False)
        with self._lock:
            self._f.write(payload + "\n")
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------------
# process tree kill (cross-platform)
# --------------------------------------------------------------------------------------------------
def _kill_tree(proc: "subprocess.Popen[str]") -> None:
    """Terminate ``proc`` and all its descendants, on POSIX and Windows."""

    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            # /T kills the whole tree, /F forces it.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
            )
        else:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError, subprocess.SubprocessError):
        # Fall back to killing just the direct child.
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _on_signal(signum: int, frame: Any) -> None:  # pragma: no cover - exercised via real signals
    proc = _ACTIVE.get("proc")
    if proc is not None:
        _kill_tree(proc)
    raise KeyboardInterrupt()


def _install_signal_handlers() -> None:
    """Best-effort: kill the child on SIGTERM/SIGINT. No-op off the main thread."""

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):  # not main thread, or unsupported
            pass


# --------------------------------------------------------------------------------------------------
# one harness invocation
# --------------------------------------------------------------------------------------------------
def _run_subprocess(
    cmd: list[str], env: dict[str, str], cwd: str, sink: TraceSink,
    deadline: float | None, cancel_path: Path,
) -> tuple[int, str | None, list[dict[str, Any]]]:
    """Launch ``cmd``, tee stdout+stderr into the trace, and watch for deadline/cancel.

    Returns ``(exit_code, reason, envelopes)`` where ``reason`` is ``None`` for a natural exit, or
    ``"timeout"`` / ``"cancelled"`` when we killed the tree. ``envelopes`` are the harness envelopes
    produced this invocation (for the caller to parse). Raises :class:`_SpawnError` if the process
    cannot start.
    """

    popen_kwargs: dict[str, Any] = dict(
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        cwd=cwd, env=env, text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except (OSError, ValueError) as exc:
        raise _SpawnError(str(exc)) from exc

    _ACTIVE["proc"] = proc
    envelopes: list[dict[str, Any]] = []
    append_lock = threading.Lock()

    def reader(stream: Any) -> None:
        try:
            for line in stream:
                env = sink.emit_harness(line)
                with append_lock:
                    envelopes.append(env)
        except Exception:
            # A dying pipe must never take the runner down.
            pass

    threads = [
        threading.Thread(target=reader, args=(proc.stdout,), daemon=True),
        threading.Thread(target=reader, args=(proc.stderr,), daemon=True),
    ]
    for t in threads:
        t.start()

    reason: str | None = None
    while proc.poll() is None:
        if deadline is not None and time.time() > deadline:
            reason = "timeout"
            break
        if cancel_path.exists():
            reason = "cancelled"
            break
        time.sleep(POLL_INTERVAL)

    if reason is not None:
        _kill_tree(proc)

    try:
        exit_code = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        exit_code = proc.returncode if proc.returncode is not None else -1
    for t in threads:
        t.join(timeout=3)
    _ACTIVE["proc"] = None
    return exit_code, reason, envelopes


# --------------------------------------------------------------------------------------------------
# driving the run
# --------------------------------------------------------------------------------------------------
def _final_nongoal_state(exit_code: int, events: list) -> str:
    """``done`` if the invocation exited cleanly or still produced a result; else ``failed``."""

    if exit_code == 0:
        return "done"
    for event in events:
        if event.kind == "turn_end" and not event.data.get("is_error") and event.data.get("result") is not None:
            return "done"
    return "failed"


def _drive(spec: dict[str, Any], harness: Harness, setup: HarnessSetup,
           status: StatusWriter, sink: TraceSink, run_dir: Path) -> int:
    goal = bool(spec.get("goal"))
    prompt = str(spec.get("prompt", ""))
    deadline = spec.get("deadline")
    deadline = float(deadline) if deadline is not None else None
    cancel_path = run_dir / "cancel"

    env = {**os.environ, **setup.env}
    cwd = setup.workdir

    all_events: list = []
    session_id: str | None = None
    goal_state: GoalState = "none"
    no_progress = 0
    attempt = 0

    sink.emit_runner({"event": "start", "harness": harness.name, "goal": goal, "deadline": deadline})

    while True:
        # Honour cancel / deadline before spending another invocation.
        if cancel_path.exists():
            status.set(state="cancelled", goal_state=goal_state, session_id=session_id)
            sink.emit_runner({"event": "done", "state": "cancelled"})
            return 0
        if deadline is not None and time.time() > deadline:
            status.set(state="timeout", goal_state=goal_state, session_id=session_id)
            sink.emit_runner({"event": "timeout", "state": "timeout"})
            return 0

        attempt += 1
        resume = session_id if attempt > 1 else None
        message = prompt if attempt == 1 else harness.continuation_prompt(goal=goal)

        try:
            cmd = harness.headless_argv(setup, message, goal=goal, resume=resume)
        except Exception as exc:
            status.set(state="failed", error=f"headless_argv failed: {exc}")
            sink.emit_runner({"event": "done", "state": "failed", "error": str(exc)})
            return 1

        sink.emit_runner({
            "event": "resume" if resume else "attempt",
            "attempt": attempt, "resume": resume,
        })
        status.set(state="running", turns=attempt, session_id=session_id, goal_state=goal_state)

        # Progress toward a goal is measured by distinct actions (see _distinct_actions): narrating
        # or re-running the same verification command does not count, so a settled goal terminates
        # while an impossible one (new actions each turn) runs until the deadline.
        progress_before = _distinct_actions(all_events)
        try:
            exit_code, reason, envelopes = _run_subprocess(cmd, env, cwd, sink, deadline, cancel_path)
        except _SpawnError as exc:
            status.set(state="failed", error=f"could not start harness: {exc}", exit_code=None)
            sink.emit_runner({"event": "done", "state": "failed", "error": str(exc)})
            return 1

        # Parse this invocation's envelopes; never let one bad line crash the runner.
        for env_obj in envelopes:
            try:
                all_events.extend(harness.parse(env_obj))
            except Exception:
                continue
        session_id = harness.session_id(all_events) or session_id
        try:
            goal_state = harness.goal_state(setup, all_events, session_id)
        except Exception:
            pass  # keep the previous goal_state on a flaky evaluator

        sink.emit_runner({"event": "exit", "attempt": attempt, "exit_code": exit_code, "reason": reason})
        sink.emit_runner({"event": "goal_check", "goal_state": goal_state, "session_id": session_id})
        status.set(turns=attempt, session_id=session_id, goal_state=goal_state, exit_code=exit_code)

        # The child was killed by us -> terminal immediately.
        if reason == "timeout":
            status.set(state="timeout")
            sink.emit_runner({"event": "timeout", "state": "timeout"})
            return 0
        if reason == "cancelled":
            status.set(state="cancelled")
            sink.emit_runner({"event": "done", "state": "cancelled"})
            return 0

        # Non-goal: a single invocation decides the outcome.
        if not goal:
            final = _final_nongoal_state(exit_code, all_events)
            error = None if final == "done" else f"harness exited with code {exit_code}"
            status.set(state=final, error=error)
            sink.emit_runner({"event": "done", "state": final})
            return 0 if final == "done" else 1

        # Goal mode.
        if goal_state == "complete":
            status.set(state="done")
            sink.emit_runner({"event": "done", "state": "done", "goal_state": "complete"})
            return 0
        if goal_state == "blocked":
            status.set(state="done")
            sink.emit_runner({"event": "done", "state": "done", "goal_state": "blocked"})
            return 0

        # No session to resume from -> we cannot continue the goal.
        if session_id is None:
            final = _final_nongoal_state(exit_code, all_events)
            status.set(state=final,
                       error=None if final == "done" else "no resumable session to continue the goal")
            sink.emit_runner({"event": "done", "state": final})
            return 0 if final == "done" else 1

        # No-progress guard (the deadline may be None for an open-ended goal).
        progress_after = _distinct_actions(all_events)
        no_progress = no_progress + 1 if progress_after <= progress_before else 0
        if no_progress >= MAX_NO_PROGRESS:
            status.set(state="done", goal_state=goal_state)
            sink.emit_runner({"event": "done", "state": "done", "reason": "no-progress"})
            return 0
        # Otherwise loop and resume toward the goal.


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.stderr.write("usage: python -m agents_on_modal.runner <run_dir>\n")
        return 2

    run_dir = Path(args[0])
    status: StatusWriter | None = None
    sink: TraceSink | None = None
    _install_signal_handlers()

    try:
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
        harness = harness_registry.get(spec["harness"])
        setup = HarnessSetup.from_json(spec["setup"])
        status = StatusWriter(run_dir)
        sink = TraceSink(run_dir)
    except Exception as exc:
        # Setup failed: record what we can so a reader is not left on "starting" forever.
        try:
            if status is None:
                status = StatusWriter(run_dir)
            status.set(state="failed", error=f"runner setup failed: {exc}")
            if sink is not None:
                sink.emit_runner({"event": "done", "state": "failed", "error": str(exc)})
        except Exception:
            pass
        if sink is not None:
            sink.close()
        return 1

    try:
        return _drive(spec, harness, setup, status, sink, run_dir)
    except (KeyboardInterrupt, SystemExit):
        proc = _ACTIVE.get("proc")
        if proc is not None:
            _kill_tree(proc)
        status.set(state="cancelled", error="runner received a termination signal")
        sink.emit_runner({"event": "done", "state": "cancelled"})
        return 0
    except Exception as exc:
        proc = _ACTIVE.get("proc")
        if proc is not None:
            _kill_tree(proc)
        status.set(state="failed", error=f"runner crashed: {exc}")
        sink.emit_runner({"event": "done", "state": "failed", "error": str(exc)})
        return 1
    finally:
        sink.close()


if __name__ == "__main__":
    raise SystemExit(main())
