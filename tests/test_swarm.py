"""Offline unit tests for ``agents_on_modal.swarm`` and the dashboard assets.

No Modal. ``models.*`` and ``sandbox.create`` / ``sandbox.list_sandboxes`` are patched to fakes, so
``Swarm.launch`` is exercised for its fan-out (N sandboxes, N runs), its model pre-warm
(``models.scale``), and ``Swarm.reconnect`` for stateless rediscovery. The dashboard files are
smoke-tested for existence and the endpoints they wire together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agents_on_modal.models.deploy as model_ops
from agents_on_modal import models, sandbox, swarm
from agents_on_modal.config import AomError
from agents_on_modal.swarm import Agent, Swarm

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "swarm"


# --------------------------------------------------------------------------------------------- #
# fakes                                                                                         #
# --------------------------------------------------------------------------------------------- #
class FakeModel:
    def __init__(self, name="m"):
        self.name = name


class FakeEndpoint:
    def __init__(self):
        self.model = FakeModel()
        self.url = "https://x.modal.direct"
        self.token = "wk-a.ws-b"


class FakeRun:
    def __init__(self, state="running", spec=None, run_id="run-1"):
        self._state = state
        self.spec = spec or {"model": "m"}
        self.run_id = run_id

    def status(self):
        return {"state": self._state}


class FakeBox:
    kind = "fake"
    _n = 0

    def __init__(self, run=None, runs=None, labels=None, box_id=None):
        FakeBox._n += 1
        self.id = box_id or f"box-{FakeBox._n}"
        self._run = run
        self._runs = runs
        self._labels = labels or {}
        self.started = None
        self.stopped = False

    def start(self, harness, endpoint, prompt, **kw):
        self.started = dict(harness=harness, endpoint=endpoint, prompt=prompt, **kw)
        self._run = self._run or FakeRun(run_id=f"{self.id}-run")
        return self._run

    def runs(self):
        if self._runs is not None:
            return list(self._runs)
        return [self._run] if self._run else []

    def labels(self):
        return dict(self._labels)

    def stop(self):
        self.stopped = True


@pytest.fixture
def launch_env(monkeypatch):
    bag = {"deployed": [], "scaled": [], "boxes": [], "endpoint": FakeEndpoint()}
    monkeypatch.setattr(models, "get", lambda n: FakeModel(n if isinstance(n, str) else n.name))
    monkeypatch.setattr(model_ops, "deploy", lambda m: (bag["deployed"].append(m), bag["endpoint"])[1])
    monkeypatch.setattr(model_ops, "scale", lambda m, **s: bag["scaled"].append(s))
    monkeypatch.setattr(sandbox, "new_id", lambda prefix="": f"{prefix}test")

    def fake_create(kind, **opts):
        box = FakeBox()
        box.create_opts = {"kind": kind, **opts}
        bag["boxes"].append(box)
        return box

    monkeypatch.setattr(sandbox, "create", fake_create)
    return bag


# --------------------------------------------------------------------------------------------- #
# launch                                                                                        #
# --------------------------------------------------------------------------------------------- #
def test_launch_creates_n_sandboxes_and_runs(launch_env):
    s = Swarm.launch("claude", "m", "build it", count=3)
    assert isinstance(s, Swarm)
    assert s.id == "swarm-test"
    assert s.model == "m"
    assert s.shared_volume == "aom-shared-swarm-test"
    assert len(launch_env["boxes"]) == 3
    assert len(s.agents()) == 3
    # every box was started with the right harness/prompt + swarm meta
    for box in launch_env["boxes"]:
        assert box.started["harness"] == "claude"
        assert box.started["prompt"] == "build it"
        assert box.started["meta"]["swarm_id"] == "swarm-test"
        assert box.create_opts["kind"] == "modal"
        assert box.create_opts["swarm"] == "swarm-test"
        assert box.create_opts["shared_volume"] == "aom-shared-swarm-test"


def test_launch_prewarms_model(launch_env):
    Swarm.launch("codex", "m", "p", count=10)
    assert len(launch_env["deployed"]) == 1
    assert launch_env["scaled"] == [{"min_containers": 4, "max_containers": 10}]


def test_launch_small_count_prewarm(launch_env):
    Swarm.launch("codex", "m", "p", count=2)
    assert launch_env["scaled"] == [{"min_containers": 2, "max_containers": 4}]


def test_launch_rejects_zero_count(launch_env):
    with pytest.raises(AomError):
        Swarm.launch("claude", "m", "p", count=0)


def test_launch_retries_then_fails(monkeypatch, launch_env):
    monkeypatch.setattr(swarm, "RETRY_BACKOFF", 0)
    calls = {"n": 0}

    def flaky_create(kind, **opts):
        calls["n"] += 1
        raise sandbox.SandboxError("no capacity")

    monkeypatch.setattr(sandbox, "create", flaky_create)
    with pytest.raises(AomError):
        Swarm.launch("claude", "m", "p", count=1)
    assert calls["n"] == swarm.CREATE_RETRIES


def test_launch_honors_explicit_shared_volume(launch_env):
    s = Swarm.launch("claude", "m", "p", count=1, shared_volume="my-vol")
    assert s.shared_volume == "my-vol"
    assert launch_env["boxes"][0].create_opts["shared_volume"] == "my-vol"


# --------------------------------------------------------------------------------------------- #
# reconnect (stateless rediscovery)                                                            #
# --------------------------------------------------------------------------------------------- #
def test_reconnect_rediscovers_live_sandboxes(monkeypatch):
    boxes = [
        FakeBox(run=FakeRun(spec={"model": "qwen"}, run_id="r0"), labels={"idx": "0"}, box_id="b0"),
        FakeBox(run=FakeRun(spec={"model": "qwen"}, run_id="r1"), labels={"idx": "1"}, box_id="b1"),
    ]
    captured = {}

    def fake_list(kind, **filters):
        captured.update(kind=kind, **filters)
        return boxes

    monkeypatch.setattr(sandbox, "list_sandboxes", fake_list)
    s = Swarm.reconnect("swarm-abc")
    assert captured == {"kind": "modal", "swarm": "swarm-abc"}
    assert s.id == "swarm-abc"
    assert s.model == "qwen"
    assert s.shared_volume == "aom-shared-swarm-abc"
    agents = s.agents()
    assert [a.id for a in agents] == ["b0", "b1"]
    assert [a.run_id for a in agents] == ["r0", "r1"]
    assert agents[0].state() == "running"


def test_reconnect_handles_sandbox_without_run(monkeypatch):
    box = FakeBox(runs=[], box_id="empty")
    monkeypatch.setattr(sandbox, "list_sandboxes", lambda kind, **f: [box])
    s = Swarm.reconnect("swarm-none")
    agent = s.agents()[0]
    assert agent.run is None
    assert agent.run_id is None
    assert agent.state() == "unknown"


# --------------------------------------------------------------------------------------------- #
# shared volume + stop                                                                          #
# --------------------------------------------------------------------------------------------- #
def test_read_and_list_shared(monkeypatch):
    class FakeEntry:
        def __init__(self, path):
            self.path = path

    class FakeVolume:
        def listdir(self, path, recursive=True):
            return [FakeEntry("/a.txt"), FakeEntry("/b.txt")]

        def read_file(self, path):
            yield b"hello "
            yield b"world"

    import types
    fake_modal = types.SimpleNamespace(Volume=types.SimpleNamespace(
        from_name=lambda name, version=2: FakeVolume()))
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)

    s = Swarm(id="s1", model="m", shared_volume="aom-shared-s1")
    assert s.list_shared() == ["/a.txt", "/b.txt"]
    assert s.read_shared("/a.txt") == b"hello world"


def test_stop_terminates_agents_and_resets_scale(monkeypatch):
    boxes = [FakeBox(box_id="x0"), FakeBox(box_id="x1")]
    scaled = []
    monkeypatch.setattr(models, "get", lambda n: FakeModel(n))
    monkeypatch.setattr(model_ops, "scale", lambda m, **s: scaled.append(s))
    s = Swarm(id="s1", model="m", shared_volume="v",
              _agents=[Agent(sandbox=b) for b in boxes])
    s.stop()
    assert all(b.stopped for b in boxes)
    assert scaled == [{"min_containers": 0}]


# --------------------------------------------------------------------------------------------- #
# dashboard assets                                                                              #
# --------------------------------------------------------------------------------------------- #
def test_index_html_exists_and_wires_endpoints():
    html = (EXAMPLES / "index.html").read_text(encoding="utf-8")
    assert "/api/agents" in html
    assert "/api/events" in html
    assert "/api/spawn" in html
    # the spawn form controls required by the task
    assert 'id="goal"' in html and 'type="checkbox"' in html
    assert 'id="timeout"' in html
    assert 'id="count"' in html
    assert 'id="prompt"' in html


def test_server_py_uses_reconnect_and_serves_index():
    src = (EXAMPLES / "server.py").read_text(encoding="utf-8")
    assert "Swarm.reconnect" in src
    assert "/api/agents" in src
    assert "/api/events" in src
    assert "/api/spawn" in src
    assert "index.html" in src
