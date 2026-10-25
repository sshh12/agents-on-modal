"""Offline, host-only tests for ``agents_on_modal.sandbox.local.LocalSandbox``.

No Docker, no Modal, no real claude/codex and no network. The primitive tests run trivial host
processes (the current Python interpreter). The end-to-end test drives ``box.start()`` plus the real
in-sandbox runner against a *fake* harness whose headless invocation just prints scripted JSON and
exits, so the whole run completes locally with no model.

Because ``Sandbox.start`` spawns the runner as a detached subprocess (a fresh interpreter), the fake
harness must be importable and registered there too. We write a tiny ``aom_fake_harness`` module and a
``sitecustomize`` that registers it into ``agents_on_modal.harness.HARNESSES`` at interpreter startup,
put both on ``PYTHONPATH`` (alongside the repo root, since aom is not pip-installed here), and let the
detached runner pick them up.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import agents_on_modal.harness as harness_registry
from agents_on_modal.harness.base import ModelAccess, SandboxProfile
from agents_on_modal.sandbox import SandboxError
from agents_on_modal.sandbox.local import LocalSandbox

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------------------
@pytest.fixture()
def aom_home(tmp_path, monkeypatch):
    """Point AOM_HOME at a throwaway dir so local state never touches the real one."""

    home = tmp_path / "aom-home"
    monkeypatch.setenv("AOM_HOME", str(home))
    return home


@pytest.fixture()
def workdir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


def _py_print(text: str) -> list[str]:
    """A portable 'echo': run this interpreter to print ``text`` (works on Windows too)."""

    return [sys.executable, "-c", f"import sys; sys.stdout.write({text!r})"]


def _fake_endpoint() -> SimpleNamespace:
    """A stand-in Endpoint with the attributes ModelAccess.from_endpoint reads."""

    model = SimpleNamespace(
        name="fake-model", context_window=131072, max_output_tokens=32000,
        reasoning=SimpleNamespace(default="high"), vision=False,
    )
    return SimpleNamespace(model=model, url="http://127.0.0.1:65500", token="fake-token-123")


# --------------------------------------------------------------------------------------------------
# lifecycle + primitives
# --------------------------------------------------------------------------------------------------
def test_create_sets_attrs_and_layout(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    assert box.kind == "local"
    assert box.id.startswith("local-")
    assert box.workdir == os.path.abspath(str(workdir))
    assert box.posix == (os.name != "nt")
    assert box.python == (sys.executable,)
    # home = <AOM_HOME>/local/<id>/home ; runs_root = <that>/runs
    assert Path(box.home) == aom_home / "local" / box.id / "home"
    assert Path(box.runs_root) == Path(box.home) / "runs"
    assert Path(box.runs_root).is_dir()
    assert (aom_home / "local" / box.id / "meta.json").is_file()
    assert box.labels() == {"kind": "local", "workdir": box.workdir}


def test_create_defaults_to_cwd(aom_home, workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    box = LocalSandbox.create()
    assert box.workdir == os.path.abspath(str(workdir))


def test_exec_runs_and_captures_stdout(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    result = box.exec(_py_print("hello-local"))
    assert result.returncode == 0
    assert result.stdout == "hello-local"
    assert result.stderr == ""


def test_exec_nonzero_and_stderr(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    result = box.exec([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(7)"])
    assert result.returncode == 7
    assert "boom" in result.stderr
    with pytest.raises(SandboxError):
        result.check("thing")


def test_exec_cwd_defaults_to_workdir(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    result = box.exec([sys.executable, "-c", "import os,sys; sys.stdout.write(os.getcwd())"])
    assert Path(result.stdout) == workdir


def test_exec_timeout_raises(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    with pytest.raises(SandboxError):
        box.exec([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.3)


def test_exec_missing_binary_raises(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    with pytest.raises(SandboxError):
        box.exec(["this_binary_does_not_exist_aom_xyz"])


def test_write_read_roundtrip_text_and_bytes(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    nested = str(workdir / "a" / "b" / "file.txt")
    box.write_file(nested, "line one\nline two\n")
    assert box.read_file(nested) == b"line one\nline two\n"

    binpath = str(workdir / "blob.bin")
    box.write_file(binpath, b"\x00\x01\x02ABC")
    assert box.read_file(binpath) == b"\x00\x01\x02ABC"


def test_read_file_offset(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    p = str(workdir / "offset.txt")
    box.write_file(p, "0123456789")
    assert box.read_file(p, offset=4) == b"456789"
    assert box.read_file(p, offset=10) == b""


def test_read_file_missing_raises_filenotfound(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    with pytest.raises(FileNotFoundError):
        box.read_file(str(workdir / "nope.txt"))


def test_spawn_is_detached_and_writes_file(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    marker = workdir / "spawned.txt"
    script = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ok', encoding='utf-8')"
    box.spawn([sys.executable, "-c", script])
    # Poll briefly: the spawned process runs independently of this call.
    deadline = time.time() + 10
    while time.time() < deadline and not marker.exists():
        time.sleep(0.05)
    assert marker.read_text(encoding="utf-8") == "ok"


def test_spawn_env_merges_over_environ(aom_home, workdir, monkeypatch):
    monkeypatch.setenv("AOM_BASE_VAR", "base")
    box = LocalSandbox.create(workdir=str(workdir))
    out = workdir / "env.txt"
    script = (
        "import os, pathlib; "
        f"pathlib.Path({str(out)!r}).write_text(os.environ.get('AOM_BASE_VAR','')+':'+os.environ.get('AOM_EXTRA',''), encoding='utf-8')"
    )
    box.spawn([sys.executable, "-c", script], env={"AOM_EXTRA": "extra"})
    deadline = time.time() + 10
    while time.time() < deadline and not out.exists():
        time.sleep(0.05)
    assert out.read_text(encoding="utf-8") == "base:extra"


def test_attach_inherits_stdio_and_returns_code(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    assert box.attach([sys.executable, "-c", "import sys; sys.exit(5)"]) == 5


def test_stop_removes_state_and_is_idempotent(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    state = aom_home / "local" / box.id
    assert state.is_dir()
    box.stop()
    assert not state.exists()
    box.stop()  # idempotent


# --------------------------------------------------------------------------------------------------
# rediscovery: list() / get()
# --------------------------------------------------------------------------------------------------
def test_list_and_get_rediscover(aom_home, workdir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    box_a = LocalSandbox.create(workdir=str(workdir))
    box_b = LocalSandbox.create(workdir=str(other))

    ids = {b.id for b in LocalSandbox.list()}
    assert {box_a.id, box_b.id} <= ids

    got = LocalSandbox.get(box_a.id)
    assert got.id == box_a.id
    assert got.workdir == box_a.workdir
    assert got.home == box_a.home
    assert got.runs_root == box_a.runs_root

    # filter by workdir
    only = LocalSandbox.list(workdir=str(other))
    assert [b.id for b in only] == [box_b.id]


def test_get_unknown_raises(aom_home):
    with pytest.raises(SandboxError):
        LocalSandbox.get("local-does-not-exist")


def test_list_empty_when_no_state(aom_home):
    assert LocalSandbox.list() == []


# --------------------------------------------------------------------------------------------------
# profile / model_access
# --------------------------------------------------------------------------------------------------
def test_profile_shape(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    profile = box.profile()
    assert isinstance(profile, SandboxProfile)
    assert profile.kind == "local"
    assert profile.posix == (os.name != "nt")
    assert profile.mcp_servers == {}
    assert profile.env == {}
    assert isinstance(profile.skills, dict)
    # Local instructions say it is the host with no isolation and host networking.
    assert "host machine" in profile.instructions
    assert box.workdir in profile.instructions


def test_model_access_is_direct(aom_home, workdir):
    box = LocalSandbox.create(workdir=str(workdir))
    endpoint = _fake_endpoint()
    access = box.model_access(endpoint)
    assert isinstance(access, ModelAccess)
    # Direct gateway root (no /v1) + the user's real token; no proxy rewriting.
    assert access.base_url == "http://127.0.0.1:65500"
    assert access.token == "fake-token-123"
    assert access.model_id == "fake-model"


# --------------------------------------------------------------------------------------------------
# end-to-end: start() drives the real runner against a fake harness
# --------------------------------------------------------------------------------------------------
_FAKE_HARNESS_MODULE = '''
"""A host-only fake harness for the local-sandbox e2e test. No model, no network."""
from __future__ import annotations
import sys
from agents_on_modal.harness.base import Harness, HarnessSetup
from agents_on_modal.traces import Event

_SCRIPT = (
    "import sys\\n"
    "for line in ("
    "'{\\"type\\": \\"session\\", \\"session_id\\": \\"sess-E2E\\"}',"
    "'{\\"type\\": \\"text\\", \\"text\\": \\"hello from fake\\"}',"
    "'{\\"type\\": \\"turn_end\\", \\"result\\": \\"ok\\", \\"is_error\\": false}'"
    "):\\n"
    "    sys.stdout.write(line + chr(10)); sys.stdout.flush()\\n"
    "sys.exit(0)\\n"
)


class FakeHarness(Harness):
    name = "fake"
    protocol = "chat"
    executable = sys.executable

    def prepare(self, access, profile, *, home, workdir):
        # Nothing network-bound; no files to write.
        return HarnessSetup(harness=self.name, home=home, workdir=workdir, env={}, files={},
                            posix=profile.posix)

    def interactive_argv(self, setup):
        return [self.executable]

    def headless_argv(self, setup, prompt, *, goal, resume):
        return [sys.executable, "-c", _SCRIPT]

    def parse(self, envelope):
        data = envelope.get("data")
        if not isinstance(data, dict):
            return []
        kind = data.get("type")
        if kind == "session":
            return [Event("session", {"session_id": data.get("session_id")})]
        if kind == "text":
            return [Event("text", {"text": data.get("text", "")})]
        if kind == "turn_end":
            return [Event("turn_end", {"result": data.get("result"), "is_error": data.get("is_error", False)})]
        return []

    def goal_state(self, setup, events, session_id):
        return "none"
'''

_SITECUSTOMIZE = '''
try:
    import agents_on_modal.harness as _h
    _h.HARNESSES.setdefault("fake", "aom_fake_harness:FakeHarness")
except Exception:
    pass
'''


@pytest.fixture()
def fake_harness(tmp_path, monkeypatch):
    """Write the fake harness + a sitecustomize that registers it, and wire up the paths.

    Returns an imported FakeHarness instance for the parent process. The detached runner subprocess
    gets the same module via PYTHONPATH + sitecustomize.
    """

    boot = tmp_path / "boot"
    boot.mkdir()
    (boot / "aom_fake_harness.py").write_text(_FAKE_HARNESS_MODULE, encoding="utf-8")
    (boot / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")

    # Parent process: make the module importable here so start() can take the instance directly,
    # and register it so Run.harness (used by events()) can reconstruct it from the spec.
    monkeypatch.syspath_prepend(str(boot))
    module = importlib.import_module("aom_fake_harness")
    module = importlib.reload(module)
    monkeypatch.setitem(harness_registry.HARNESSES, "fake", "aom_fake_harness:FakeHarness")

    # Detached runner subprocess: it is a fresh interpreter launched with cwd = workdir, and aom is
    # not pip-installed, so it needs the repo root (for agents_on_modal) and boot (fake + sitecustomize).
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT), str(boot)] + ([existing] if existing else [])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parts))

    yield module.FakeHarness()

    sys.modules.pop("aom_fake_harness", None)


def test_start_runs_to_done_and_streams_events(aom_home, workdir, fake_harness):
    box = LocalSandbox.create(workdir=str(workdir))
    endpoint = _fake_endpoint()

    run = box.start(fake_harness, endpoint, "do the thing")

    # spec.json was written before the detached runner was spawned.
    assert run.spec["harness"] == "fake"
    assert run.spec["model"] == "fake-model"

    status = run.wait(timeout=60, poll=0.1)
    assert status["state"] == "done", status
    assert status["session_id"] == "sess-E2E"

    events = list(run.events(follow=True, poll=0.05))
    kinds = [e.kind for e in events]
    assert "session" in kinds
    texts = [e.data.get("text") for e in events if e.kind == "text"]
    assert "hello from fake" in texts
    assert any(e.kind == "turn_end" and not e.data.get("is_error") for e in events)

    # The runner recorded a terminal "done" lifecycle event too.
    assert any(e.kind == "status" and e.data.get("state") == "done" for e in events)


def test_runs_lists_the_started_run(aom_home, workdir, fake_harness):
    box = LocalSandbox.create(workdir=str(workdir))
    run = box.start(fake_harness, _fake_endpoint(), "do the thing")
    run.wait(timeout=60, poll=0.1)

    listed = box.runs()
    assert run.run_id in [r.run_id for r in listed]
