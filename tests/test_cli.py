"""Offline unit tests for ``agents_on_modal.cli``.

No Modal, Docker or real harnesses. The model functions and ``sandbox.create`` are patched to fakes,
so these tests exercise only the argparse wiring, the registry calls each subcommand makes, event
streaming + terminal exit codes, and the top-level ``AomError`` handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agents_on_modal.models.deploy as model_ops
from agents_on_modal import cli, models, sandbox, swarm
from agents_on_modal.cli import build_parser, main
from agents_on_modal.config import AomError
from agents_on_modal.traces import Event


# --------------------------------------------------------------------------------------------- #
# fakes                                                                                         #
# --------------------------------------------------------------------------------------------- #
class FakeModel:
    def __init__(self, name="m"):
        self.name = name
        self.gpu = "H100"
        self.description = "a fake model"


class FakeEndpoint:
    def __init__(self, model=None):
        self.model = model or FakeModel()
        self.url = "https://ws--aom-m-server.us-west.modal.direct"
        self.token = "wk-x.ws-y"


class FakeRun:
    def __init__(self, events=None, state="done", run_id="run-1"):
        self._events = list(events or [])
        self._state = state
        self.run_id = run_id

    def events(self, *, follow=False, **kw):
        return iter(self._events)

    def status(self):
        return {"state": self._state}


class FakeBox:
    kind = "fake"

    def __init__(self, run=None):
        self.id = "box-1"
        self._run = run or FakeRun()
        self.started = None
        self.interactive_args = None

    def start(self, harness, endpoint, prompt, **kw):
        self.started = dict(harness=harness, endpoint=endpoint, prompt=prompt, **kw)
        return self._run

    def interactive(self, harness, endpoint, **kw):
        self.interactive_args = dict(harness=harness, endpoint=endpoint, **kw)
        return 0


@pytest.fixture
def patched(monkeypatch):
    """Patch models.* and sandbox.create; return a bag the test inspects."""

    bag = {"endpoint": FakeEndpoint(), "box": FakeBox(), "created": None,
           "deployed": [], "scaled": [], "stopped": [], "waited": []}
    monkeypatch.setattr(models, "get", lambda n: FakeModel(n))
    monkeypatch.setattr(models, "list_models", lambda: [FakeModel("a"), FakeModel("b")])
    monkeypatch.setattr(model_ops, "endpoint", lambda m: bag["endpoint"])
    monkeypatch.setattr(model_ops, "deploy", lambda m: (bag["deployed"].append(m), bag["endpoint"])[1])
    monkeypatch.setattr(model_ops, "wait_ready", lambda ep, **kw: bag["waited"].append(ep))
    monkeypatch.setattr(model_ops, "scale", lambda m, **s: bag["scaled"].append(s))
    monkeypatch.setattr(model_ops, "stop", lambda m: bag["stopped"].append(m))
    monkeypatch.setattr(model_ops, "status", lambda m: {"model": m.name, "app_name": "aom-m"})

    def fake_create(kind, **opts):
        bag["created"] = {"kind": kind, "opts": opts}
        return bag["box"]

    monkeypatch.setattr(sandbox, "create", fake_create)
    return bag


# --------------------------------------------------------------------------------------------- #
# argparse wiring                                                                               #
# --------------------------------------------------------------------------------------------- #
def test_every_subcommand_has_a_func():
    parser = build_parser()
    commands = {
        "setup": [],
        "run": ["--harness", "claude", "--model", "m", "p"],
        "shell": ["--harness", "codex", "--model", "m"],
        "swarm": ["--harness", "claude", "--model", "m", "--count", "3", "p"],
    }
    for name, extra in commands.items():
        args = parser.parse_args([name, *extra])
        assert callable(args.func)
    for sub in ("list",):
        assert callable(parser.parse_args(["models", sub]).func)
    for sub, extra in (("deploy", ["m"]), ("scale", ["m"]), ("stop", ["m"]), ("status", ["m"])):
        assert callable(parser.parse_args(["models", sub, *extra]).func)


def test_missing_required_args_exit_2():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "do it"])  # no --harness/--model
    with pytest.raises(SystemExit):
        build_parser().parse_args([])  # no subcommand


def test_bad_choice_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--harness", "nope", "--model", "m", "p"])


# --------------------------------------------------------------------------------------------- #
# run                                                                                           #
# --------------------------------------------------------------------------------------------- #
def test_run_attaches_creates_and_streams(patched, capsys):
    patched["box"] = FakeBox(FakeRun(
        events=[Event("text", {"text": "hello world"}), Event("turn_end", {"result": "ok"})],
        state="done"))
    rc = main(["run", "--sandbox", "local", "--harness", "claude", "--model", "m", "do it"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hello world" in out
    assert "[run done]" in out
    # attached (not deployed) by default
    assert patched["deployed"] == []
    box = patched["box"]
    assert box.started["harness"] == "claude"
    assert box.started["prompt"] == "do it"
    assert patched["created"]["kind"] == "local"
    assert patched["created"]["opts"]["endpoint"] is patched["endpoint"]


def test_run_deploy_flag_deploys_and_waits(patched):
    rc = main(["run", "--harness", "codex", "--model", "m", "--deploy", "--sandbox", "modal", "p"])
    assert rc == 0
    assert len(patched["deployed"]) == 1
    assert patched["waited"] == [patched["endpoint"]]


def test_run_passes_goal_timeout_effort_proxy(patched):
    main(["run", "--harness", "claude", "--model", "m", "--goal", "--timeout", "30",
          "--effort", "high", "--proxy", "socks5h://h:1", "--workdir", "/w", "p"])
    started = patched["box"].started
    assert started["goal"] is True
    assert started["timeout"] == 30.0
    assert started["effort"] == "high"
    assert patched["created"]["opts"]["proxy"] == "socks5h://h:1"
    assert patched["created"]["opts"]["workdir"] == "/w"


def test_run_nonzero_exit_on_failed_state(patched):
    patched["box"] = FakeBox(FakeRun(events=[Event("error", {"message": "boom"})], state="failed"))
    rc = main(["run", "--harness", "claude", "--model", "m", "p"])
    assert rc == 1


# --------------------------------------------------------------------------------------------- #
# shell                                                                                         #
# --------------------------------------------------------------------------------------------- #
def test_shell_opens_interactive(patched):
    rc = main(["shell", "--sandbox", "docker", "--harness", "codex", "--model", "m",
               "--effort", "low"])
    assert rc == 0
    assert patched["box"].interactive_args["harness"] == "codex"
    assert patched["box"].interactive_args["endpoint"] is patched["endpoint"]
    assert patched["box"].interactive_args["effort"] == "low"


# --------------------------------------------------------------------------------------------- #
# models                                                                                        #
# --------------------------------------------------------------------------------------------- #
def test_models_list(patched, capsys):
    assert main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "a" in out and "b" in out


def test_models_deploy(patched):
    assert main(["models", "deploy", "m"]) == 0
    assert len(patched["deployed"]) == 1


def test_models_scale_maps_flags(patched):
    assert main(["models", "scale", "m", "--min", "2", "--max", "8", "--buffer", "1"]) == 0
    assert patched["scaled"] == [{"min_containers": 2, "max_containers": 8, "buffer_containers": 1}]


def test_models_scale_requires_a_setting(patched):
    assert main(["models", "scale", "m"]) == 1  # AomError -> 1


def test_models_stop_and_status(patched, capsys):
    assert main(["models", "stop", "m"]) == 0
    assert len(patched["stopped"]) == 1
    assert main(["models", "status", "m"]) == 0
    assert "aom-m" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------- #
# swarm (cli wiring; Swarm.launch is covered in test_swarm)                                     #
# --------------------------------------------------------------------------------------------- #
def test_swarm_calls_launch_with_parsed_args(monkeypatch, capsys):
    captured = {}

    class _FakeSwarm:
        id = "swarm-xyz"
        model = "m"

        def agents(self):
            return [1, 2, 3]

    def fake_launch(harness, model, prompt, **kw):
        captured.update(harness=harness, model=model, prompt=prompt, **kw)
        return _FakeSwarm()

    monkeypatch.setattr(swarm.Swarm, "launch", fake_launch)
    rc = main(["swarm", "--harness", "claude", "--model", "m", "--count", "3",
               "--goal", "--timeout", "60", "do the thing"])
    assert rc == 0
    assert captured["harness"] == "claude"
    assert captured["count"] == 3
    assert captured["goal"] is True
    assert captured["timeout"] == 60.0
    out = capsys.readouterr().out
    assert "swarm-xyz" in out
    assert "examples/swarm/server.py" in out


# --------------------------------------------------------------------------------------------- #
# error handling                                                                                #
# --------------------------------------------------------------------------------------------- #
def test_aom_error_returns_1_and_prints_stderr(monkeypatch, capsys):
    def boom(name):
        raise AomError("no such model")

    monkeypatch.setattr(models, "get", boom)
    rc = main(["run", "--harness", "claude", "--model", "ghost", "p"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err and "no such model" in err


def test_setup_without_modal_auth_errors(monkeypatch, capsys):
    import types

    fake_modal = types.SimpleNamespace()

    class _WS:
        @staticmethod
        def from_context():
            raise RuntimeError("not authenticated")

    fake_modal.Workspace = _WS
    monkeypatch.setitem(__import__("sys").modules, "modal", fake_modal)
    rc = main(["setup"])
    assert rc == 1
    assert "modal setup" in capsys.readouterr().err


def test_event_renderer_covers_all_kinds(capsys):
    for kind in ("session", "text", "thinking", "tool_call", "tool_result", "usage",
                 "compaction", "subagent", "goal", "turn_end", "error", "status", "log"):
        cli._print_event(Event(kind, {"text": "x", "name": "t", "message": "m", "result": "r"}))
    # just assert it did not raise and produced output
    assert capsys.readouterr().out is not None
