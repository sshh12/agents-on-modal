"""Offline unit tests for ``agents_on_modal.swarm`` and the dashboard assets.

No Modal. ``models.*`` and ``sandbox.create`` / ``sandbox.list_sandboxes`` are patched to fakes, so
``Swarm.launch`` is exercised for its fan-out (N sandboxes, N runs), its model pre-warm
(``models.scale``), and ``Swarm.reconnect`` for stateless rediscovery. The dashboard files are
smoke-tested for existence and the endpoints they wire together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agents_on_modal.models.lifecycle as model_ops
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
        self.scaling = models.Scaling()  # defaults: target_concurrency=32.0


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
    monkeypatch.setattr(model_ops, "ensure_deployed",
                        lambda m, **k: (bag["deployed"].append((m, k)), bag["endpoint"])[1])
    monkeypatch.setattr(model_ops, "deploy", lambda *a, **k: pytest.fail("launch must not redeploy"))
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
    s = Swarm.launch("claude", "m", "build it", count=3, max_containers=1)
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


def test_launch_deploys_but_does_not_scale(launch_env):
    # The model is a shared, account-level resource: launch only ensures it is deployed
    # (idempotent). It must NOT scale per swarm -- Modal autoscales the shared endpoint from
    # aggregate load under the catalog policy; scaling here would stomp that global setting.
    Swarm.launch("codex", "m", "p", count=10, max_containers=1)
    assert len(launch_env["deployed"]) == 1
    assert launch_env["deployed"][0][1] == {"max_containers": 1}  # the swarm's ask, passed through
    assert launch_env["scaled"] == []


def test_launch_never_scales_regardless_of_count(launch_env):
    # Even a large fan-out does not provision GPUs by agent count -- no scale call at all.
    Swarm.launch("codex", "m", "p", count=100, max_containers=1)
    assert launch_env["scaled"] == []


def test_launch_rejects_zero_count(launch_env):
    with pytest.raises(AomError):
        Swarm.launch("claude", "m", "p", count=0, max_containers=1)


def test_launch_retries_then_fails(monkeypatch, launch_env):
    monkeypatch.setattr(swarm, "RETRY_BACKOFF", 0)
    calls = {"n": 0}

    def flaky_create(kind, **opts):
        calls["n"] += 1
        raise sandbox.SandboxError("no capacity")

    monkeypatch.setattr(sandbox, "create", flaky_create)
    with pytest.raises(AomError):
        Swarm.launch("claude", "m", "p", count=1, max_containers=1)
    assert calls["n"] == swarm.CREATE_RETRIES


def test_launch_honors_explicit_shared_volume(launch_env):
    s = Swarm.launch("claude", "m", "p", count=1, max_containers=1, shared_volume="my-vol")
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


def test_stop_terminates_agents_without_scaling_model(monkeypatch):
    # stop() tears down only its own sandboxes; the shared model is left to scale down on its own
    # (another consumer may be using it), so it must NOT touch the autoscaler.
    boxes = [FakeBox(box_id="x0"), FakeBox(box_id="x1")]
    scaled = []
    monkeypatch.setattr(models, "get", lambda n: FakeModel(n))
    monkeypatch.setattr(model_ops, "scale", lambda m, **s: scaled.append(s))
    s = Swarm(id="s1", model="m", shared_volume="v",
              _agents=[Agent(sandbox=b) for b in boxes])
    s.stop()
    assert all(b.stopped for b in boxes)
    assert scaled == []


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


def _server():
    import sys

    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    import server  # examples/swarm/server.py

    return server


def test_dashboard_zip_download_is_an_attachment_named_after_the_volume(monkeypatch):
    import threading
    import urllib.request

    server = _server()
    calls = []

    def fake_zip(name, dest, *, all_files=False):
        calls.append((name, all_files))
        dest.write(b"PK-zip-bytes")
        return {"files": 3}

    monkeypatch.setattr(server.export, "zip_shared_volume", fake_zip, raising=True)
    httpd = server._bind("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urllib.request.urlopen(base + "/api/zip?swarm=swarm-1&all=1") as res:
            assert res.headers["Content-Type"] == "application/zip"
            assert res.headers["Content-Disposition"] == 'attachment; filename="aom-shared-swarm-1-all.zip"'
            assert res.headers["X-Aom-Files"] == "3"
            assert res.read() == b"PK-zip-bytes"
        with urllib.request.urlopen(base + "/api/zip?swarm=swarm-1") as res:
            assert 'filename="aom-shared-swarm-1.zip"' in res.headers["Content-Disposition"]
        assert calls == [("aom-shared-swarm-1", True), ("aom-shared-swarm-1", False)]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_zip_rejects_a_bad_swarm_id():
    server = _server()
    for bad in ("", "../x", "a b", "x/y"):
        with pytest.raises(AomError):
            server._zip_export(bad, False)


def test_index_html_wires_the_zip_download():
    html = (EXAMPLES / "index.html").read_text(encoding="utf-8")
    assert 'id="zipBtn"' in html and 'id="zipScope"' in html
    assert "/api/zip?swarm=" in html


# --------------------------------------------------------------------------------------------- #
# hard stop (dashboard "stop swarm")                                                            #
# --------------------------------------------------------------------------------------------- #
class _Listing:
    """Stand-in for the sandbox module: ``list_sandboxes`` returns the given boxes."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def list_sandboxes(self, kind, **filters):
        self.calls.append((kind, filters))
        return list(self.boxes)


def test_stop_swarm_terminates_every_tagged_sandbox_even_when_one_fails(tmp_path):
    from agents_on_modal.dynamic import LocalDirControlPlane

    class Stuck(FakeBox):
        def stop(self):
            raise RuntimeError("terminate failed")

    boxes = [FakeBox(), Stuck(box_id="stuck"), FakeBox()]
    listing = _Listing(boxes)
    control = LocalDirControlPlane(str(tmp_path))  # no meta.json: a naive swarm
    out = swarm.stop_swarm("swarm-x", sandboxes=listing, control=control)
    assert listing.calls == [("modal", {"swarm": "swarm-x"})]  # found by the swarm tag
    assert boxes[0].stopped and boxes[2].stopped
    assert out == {"swarm": "swarm-x", "mode": "naive", "found": 3, "stopped": 2,
                   "errors": [{"id": "stuck", "error": "terminate failed"}]}
    assert not control.exists("_swarm/stopped.json")  # naive swarms have no controller to halt


def test_stop_swarm_on_a_dynamic_swarm_marks_it_stopped_and_cancels_live_records(tmp_path):
    from agents_on_modal.dynamic import LocalDirControlPlane

    control = LocalDirControlPlane(str(tmp_path))
    control.put_json("_swarm/meta.json", {"swarm_id": "swarm-d", "model": "m"})
    control.put_json("_swarm/agents/a.json", {"id": "a", "state": "running"})
    control.put_json("_swarm/agents/b.json", {"id": "b", "state": "done"})
    orchestrator, agent = FakeBox(box_id="orch"), FakeBox(box_id="a")
    out = swarm.stop_swarm("swarm-d", sandboxes=_Listing([orchestrator, agent]), control=control)
    assert out["mode"] == "dynamic" and out["stopped"] == 2 and orchestrator.stopped and agent.stopped
    assert control.get_json("_swarm/stopped.json")["swarm_id"] == "swarm-d"
    assert control.get_json("_swarm/agents/a.json")["state"] == "cancelled"
    assert control.get_json("_swarm/agents/b.json")["state"] == "done"  # finished work keeps its state


def test_dashboard_stop_halts_this_process_controller_then_sweeps_twice(monkeypatch):
    import threading

    server = _server()
    sweeps, halts = [], []

    class Ctl:
        def request_halt(self):
            halts.append(True)

    done = threading.Thread(target=lambda: None)
    done.start()
    monkeypatch.setitem(server._controllers, "swarm-d",
                        {"thread": done, "stop": threading.Event(), "controller": Ctl()})
    monkeypatch.setattr(server, "stop_swarm", lambda sid: (sweeps.append(sid), {
        "swarm": sid, "mode": "dynamic", "found": 3 if len(sweeps) == 1 else 1,
        "stopped": 3 if len(sweeps) == 1 else 1, "errors": []})[1])
    out = server._stop({"swarm": "swarm-d"})
    assert halts == [True] and sweeps == ["swarm-d", "swarm-d"]
    assert out["stopped"] == 4 and out["found"] == 4
    assert server._controllers["swarm-d"]["stop"].is_set()
    for bad in ("", "../x", "a b"):
        with pytest.raises(AomError):
            server._stop({"swarm": bad})


def test_dashboard_stop_endpoint_and_button_are_wired(monkeypatch):
    import json
    import threading
    import urllib.request

    server = _server()
    monkeypatch.setattr(server, "stop_swarm", lambda sid: {"swarm": sid, "mode": "naive", "found": 2,
                                                           "stopped": 2, "errors": []})
    httpd = server._bind("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/api/stop",
                                     data=json.dumps({"swarm": "swarm-n"}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            assert json.loads(res.read())["stopped"] == 2
    finally:
        httpd.shutdown()
        httpd.server_close()
    html = (EXAMPLES / "index.html").read_text(encoding="utf-8")
    assert 'id="stopBtn"' in html and 'class="danger"' in html
    assert '"/api/stop"' in html and "confirm(" in html


def test_polling_a_stopped_dynamic_swarm_never_restarts_its_controller(tmp_path, monkeypatch):
    from agents_on_modal.dynamic import LocalDirControlPlane

    server = _server()
    control = LocalDirControlPlane(str(tmp_path))
    started = []
    monkeypatch.setattr(server, "VolumeControlPlane", lambda name: control)
    monkeypatch.setattr(server.DynamicSwarm, "reconnect", classmethod(
        lambda cls, sid, **k: type("DS", (), {"id": sid, "model": "m"})()))
    monkeypatch.setattr(server, "_ensure_controller", lambda ds, **k: started.append(ds.id))

    assert server._dynamic_agents_payload("swarm-d", {"model": "m"})["stopped"] is False
    assert started == ["swarm-d"]  # a live swarm is resumed as before
    control.put_json("_swarm/stopped.json", {"ts": 1})
    assert server._dynamic_agents_payload("swarm-d", {"model": "m"})["stopped"] is True
    assert started == ["swarm-d"]  # ...but a hard-stopped one stays stopped
