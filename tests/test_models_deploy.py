"""Unit tests for deploy.py: pure name/URL derivation + mocked SDK call wiring."""

from __future__ import annotations

import json

import pytest

import agents_on_modal.models.lifecycle as deploy
from agents_on_modal.config import AomError, ProxyToken
from agents_on_modal.models import Endpoint, Model, Scaling, get


TOKEN = ProxyToken("wk-abc123", "ws-def456")


def _model() -> Model:
    return get("qwen3.8-27b")


# --------------------------------------------------------------------------- #
# conservative shared scaling policy                                           #
# --------------------------------------------------------------------------- #
def test_scaling_defaults_are_conservative_and_shared():
    # The catalog policy packs before spawning: scale to zero when idle, a long scale-up window, and
    # a high per-GPU concurrency (the sharing unit). No per-caller fan-out.
    s = Scaling()
    assert s.min_containers == 0          # idle -> zero (gateway holds the next cold start)
    assert s.buffer_containers == 0
    assert s.scaleup_window >= 30         # bursts queue instead of cold-starting a GPU
    assert s.target_concurrency >= 32.0   # many requests share one GPU


def test_container_cap_is_not_a_model_setting():
    # The cap is the apps' call (ensure_deployed / deploy max_containers=), never a catalog default.
    assert not hasattr(Scaling(), "max_containers")
    with pytest.raises(AomError, match="max_containers is not a catalog setting"):
        Model.from_json({**_model().to_json(), "scaling": {"max_containers": 1}})


def test_scaling_round_trips_through_model_json():
    # The whole policy (incl. the new scaleup_window) survives Model.to_json/from_json so it is
    # baked into AOM_MODEL_JSON and applied at deploy. Catalog partial overrides keep the defaults.
    m = get("qwen3.8-27b")
    assert m.scaling == Model.from_json(m.to_json()).scaling
    over = Model.from_json({**m.to_json(), "scaling": {"target_concurrency": 4.0}})
    assert over.scaling.target_concurrency == 4.0
    assert over.scaling.scaledown_window == Scaling().scaledown_window  # inherits the default


# --------------------------------------------------------------------------- #
# pure derivation                                                              #
# --------------------------------------------------------------------------- #
def test_shorten_app_name_short_name_unchanged():
    assert deploy.shorten_app_name("aom-qwen3-8-27b", "myws") == "aom-qwen3-8-27b"


def test_shorten_app_name_long_name_is_deterministic_and_fits():
    workspace = "my-workspace"
    long_name = "aom-" + "x" * 80
    short = deploy.shorten_app_name(long_name, workspace)
    # fits inside the DNS label
    label = f"{workspace}--{short}-server"
    assert len(label) <= deploy.MAX_DNS_LABEL_LENGTH
    # deterministic + ends in the 8-char digest
    assert deploy.shorten_app_name(long_name, workspace) == short
    assert short != long_name
    assert len(short.rsplit("-", 1)[-1]) == 8


def test_shorten_app_name_rejects_overlong_workspace():
    with pytest.raises(AomError):
        deploy.shorten_app_name("aom-x", "w" * 60)


def test_base_url_shape():
    url = deploy.base_url("aom-qwen3-8-27b", "myws", "us-west")
    assert url == "https://myws--aom-qwen3-8-27b-server.us-west.modal.direct"
    assert "/v1" not in url  # Endpoint.openai_base_url appends /v1


def test_base_url_rejects_overlong_label():
    with pytest.raises(AomError):
        deploy.base_url("a" * 70, "myws", "us-west")


def test_deployed_app_name_uses_model_app_name():
    model = _model()
    assert deploy.deployed_app_name(model, "myws") == model.app_name == "aom-qwen3-8-27b"


def test_endpoint_url_round_trips_into_openai_base_url():
    url = deploy.base_url("aom-qwen3-8-27b", "myws", "us-west")
    endpoint = Endpoint(model=_model(), url=url, token=TOKEN.bearer)
    assert endpoint.openai_base_url == url + "/v1"


# --------------------------------------------------------------------------- #
# deploy() wiring (SDK seams mocked)                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture
def seams(monkeypatch):
    """Mock every Modal seam of deploy/ensure_deployed; ``calls`` records them in order."""

    calls: list[tuple] = []
    state = {"deployed": True, "live_max": 1, "env": None}
    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")

    def fake_modal_deploy(name, env):
        state["env"] = env
        calls.append(("deploy", name))

    monkeypatch.setattr(deploy, "_modal_deploy", fake_modal_deploy)
    monkeypatch.setattr(deploy, "_run_prepare", lambda name: calls.append(("prepare", name)))
    monkeypatch.setattr(deploy, "_server_deployed", lambda name: state["deployed"])
    monkeypatch.setattr(deploy, "_live_max_containers", lambda name: state["live_max"])
    monkeypatch.setattr(deploy, "_set_max_containers", lambda name, n: calls.append(("cap", n)))
    return calls, state


def test_deploy_calls_seams_with_right_args(seams):
    calls, state = seams
    model = _model()

    endpoint = deploy.deploy(model, max_containers=3, token=TOKEN)

    # the cap is pinned live right after the deploy; weights prepared AFTER the app is deployed
    assert calls == [("deploy", "aom-qwen3-8-27b"), ("cap", 3), ("prepare", "aom-qwen3-8-27b")]
    env = state["env"]

    # env carries a round-trippable model spec, the cap, + the pinned builder version
    assert env[deploy.AOM_APP_NAME_ENV] == "aom-qwen3-8-27b"
    assert env[deploy.AOM_MAX_CONTAINERS_ENV] == "3"
    assert env["MODAL_IMAGE_BUILDER_VERSION"] == deploy.MODAL_IMAGE_BUILDER_VERSION
    round_tripped = Model.from_json(json.loads(env[deploy.AOM_MODEL_JSON_ENV]))
    assert round_tripped.name == model.name
    assert round_tripped.engine == model.engine

    assert isinstance(endpoint, Endpoint)
    assert endpoint.url == "https://myws--aom-qwen3-8-27b-server.us-west.modal.direct"
    assert endpoint.token == TOKEN.bearer
    assert endpoint.model is model


def test_deploy_scaling_override_reaches_env(seams):
    _, state = seams

    deploy.deploy(_model(), max_containers=16, scaling=Scaling(min_containers=4), token=TOKEN)

    scaling = json.loads(state["env"][deploy.AOM_MODEL_JSON_ENV])["scaling"]
    assert scaling["min_containers"] == 4
    assert "max_containers" not in scaling  # the cap travels on its own, not in the model spec
    assert state["env"][deploy.AOM_MAX_CONTAINERS_ENV] == "16"


@pytest.mark.parametrize("bad", [0, -1, None, "2", 1.5, True])
def test_deploy_and_ensure_reject_a_bad_cap(seams, bad):
    calls, _ = seams
    for fn in (deploy.deploy, deploy.ensure_deployed):
        with pytest.raises(AomError, match="max_containers must be a positive integer"):
            fn(_model(), max_containers=bad, token=TOKEN)
    assert calls == []


def test_endpoint_attaches_without_deploying(monkeypatch):
    model = _model()
    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    # if it tried to deploy, these would blow up
    monkeypatch.setattr(deploy, "_modal_deploy", lambda *a, **k: pytest.fail("should not deploy"))

    endpoint = deploy.endpoint(model, token=TOKEN)
    assert endpoint.url == "https://myws--aom-qwen3-8-27b-server.us-west.modal.direct"
    assert endpoint.token == TOKEN.bearer


def test_ensure_deployed_attaches_to_a_live_app_without_redeploying(seams):
    # A redeploy would replace the running containers under every other app on the endpoint, so a
    # deployed app at (or above) the asked-for cap is attached as-is; only the resumable prepare runs.
    calls, _ = seams

    endpoint = deploy.ensure_deployed(_model(), max_containers=1, token=TOKEN)

    assert calls == [("prepare", "aom-qwen3-8-27b")]
    assert endpoint.url == "https://myws--aom-qwen3-8-27b-server.us-west.modal.direct"
    assert endpoint.token == TOKEN.bearer


def test_ensure_deployed_raises_a_live_cap_without_redeploying(seams):
    # The n=1 swarm is running; a second app asks for 2 -> the cap goes up live, nothing restarts.
    calls, state = seams
    state["live_max"] = 1

    deploy.ensure_deployed(_model(), max_containers=2, token=TOKEN)

    assert calls == [("prepare", "aom-qwen3-8-27b"), ("cap", 2)]


def test_ensure_deployed_never_lowers_a_live_cap(seams):
    # Another app asked for 3; an app asking for 1 must not take that away (only scale() lowers).
    calls, state = seams
    state["live_max"] = 3

    deploy.ensure_deployed(_model(), max_containers=1, token=TOKEN)

    assert ("cap", 1) not in calls


def test_ensure_deployed_caps_an_endpoint_reporting_no_cap(seams):
    # Modal reports None when no Server-level cap is in effect: establish the asked-for one.
    calls, state = seams
    state["live_max"] = None

    deploy.ensure_deployed(_model(), max_containers=1, token=TOKEN)

    assert ("cap", 1) in calls


def test_ensure_deployed_deploys_at_the_asked_cap_when_not_deployed(seams):
    calls, state = seams
    state["deployed"] = False

    endpoint = deploy.ensure_deployed(_model(), max_containers=2, token=TOKEN)

    assert calls == [("deploy", "aom-qwen3-8-27b"), ("cap", 2), ("prepare", "aom-qwen3-8-27b")]
    assert state["env"][deploy.AOM_MAX_CONTAINERS_ENV] == "2"
    assert endpoint.url == "https://myws--aom-qwen3-8-27b-server.us-west.modal.direct"


def test_ensure_deployed_serializes_concurrent_callers_per_app(seams, monkeypatch):
    # Two apps in one process racing on an undeployed model: exactly one deploys, the other attaches.
    import threading

    calls, state = seams
    state["deployed"] = False
    gate = threading.Event()

    def slow_deploy(name, env):
        gate.wait(5)
        calls.append(("deploy", name))
        state["deployed"] = True

    monkeypatch.setattr(deploy, "_modal_deploy", slow_deploy)
    threads = [threading.Thread(target=deploy.ensure_deployed, args=(_model(),),
                                kwargs={"max_containers": 1, "token": TOKEN}) for _ in range(2)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)

    assert [c for c in calls if c[0] == "deploy"] == [("deploy", "aom-qwen3-8-27b")]


# --------------------------------------------------------------------------- #
# scale() / stop()                                                             #
# --------------------------------------------------------------------------- #
class _FakeAutoscaler:
    def __init__(self):
        self.updated: dict | None = None

    def update(self, **kwargs):
        self.updated = kwargs


class _FakeServer:
    def __init__(self):
        self.autoscaler = _FakeAutoscaler()


def test_scale_updates_autoscaler(monkeypatch):
    import modal

    fake = _FakeServer()
    seen = {}

    def from_name(app_name, name):
        seen["app"] = app_name
        seen["cls"] = name
        return fake

    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    monkeypatch.setattr(modal.Server, "from_name", staticmethod(from_name))

    deploy.scale(_model(), min_containers=4, max_containers=32)

    assert seen == {"app": "aom-qwen3-8-27b", "cls": "Server"}
    assert fake.autoscaler.updated == {"min_containers": 4, "max_containers": 32}


def test_scale_rejects_unknown_setting(monkeypatch):
    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    with pytest.raises(AomError):
        deploy.scale(_model(), bogus=1)


def test_scale_requires_a_setting(monkeypatch):
    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    with pytest.raises(AomError):
        deploy.scale(_model())


def test_scale_falls_back_to_update_autoscaler(monkeypatch):
    import modal

    class LegacyServer:  # modal 1.5.x: no .autoscaler, has update_autoscaler
        def __init__(self):
            self.legacy = None

        def update_autoscaler(self, **kwargs):
            self.legacy = kwargs

    legacy = LegacyServer()
    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    monkeypatch.setattr(modal.Server, "from_name", staticmethod(lambda a, n: legacy))

    deploy.scale(_model(), target_concurrency=8.0)
    assert legacy.legacy == {"target_concurrency": 8.0}


def test_stop_calls_app_stop(monkeypatch):
    import modal

    class FakeApp:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    fake = FakeApp()
    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    monkeypatch.setattr(modal.App, "lookup", staticmethod(lambda name: fake))

    deploy.stop(_model())
    assert fake.stopped is True


def test_stop_without_app_stop_uses_the_cli(monkeypatch):
    # modal 1.6 has no App.stop and rejects max_containers=0 (the old fallback raised InvalidError).
    import modal

    stopped: list[str] = []
    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    monkeypatch.setattr(modal.App, "lookup", staticmethod(lambda name: object()))
    monkeypatch.setattr(deploy, "_modal_app_stop", lambda name: stopped.append(name))
    deploy.stop(_model())
    assert stopped == ["aom-qwen3-8-27b"]


def test_stop_is_idempotent_when_app_missing(monkeypatch):
    import modal

    def missing(name):
        raise RuntimeError("App not found")

    monkeypatch.setattr(deploy, "_workspace_slug", lambda: "myws")
    monkeypatch.setattr(modal.App, "lookup", staticmethod(missing))
    deploy.stop(_model())  # no raise


# --------------------------------------------------------------------------- #
# wait_ready()                                                                 #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status):
        self.status = status

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_wait_ready_returns_on_200(monkeypatch):
    seen = {}

    def urlopen(request, timeout=0):
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = request.full_url
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    endpoint = Endpoint(model=_model(), url="https://x.modal.direct", token=TOKEN.bearer)
    deploy.wait_ready(endpoint, timeout=5)

    assert seen["url"] == "https://x.modal.direct/health"
    assert seen["auth"] == f"Bearer {TOKEN.bearer}"


def test_wait_ready_raises_on_auth_failure(monkeypatch):
    import urllib.error

    def urlopen(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    endpoint = Endpoint(model=_model(), url="https://x.modal.direct", token=TOKEN.bearer)
    with pytest.raises(AomError):
        deploy.wait_ready(endpoint, timeout=5)


def test_wait_ready_times_out(monkeypatch):
    import urllib.error

    def urlopen(request, timeout=0):
        raise urllib.error.URLError("refused")  # cold-starting

    clock = {"t": 0.0}
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    monkeypatch.setattr(deploy.time, "monotonic", lambda: clock.__setitem__("t", clock["t"] + 1) or clock["t"])

    endpoint = Endpoint(model=_model(), url="https://x.modal.direct", token=TOKEN.bearer)
    with pytest.raises(AomError):
        deploy.wait_ready(endpoint, timeout=3)
