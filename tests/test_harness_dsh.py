"""Unit tests for the DeepSeek Harness (`dsh`) harness and its SDK adapter.

Offline: no real `dsh`, no `deepseek_harness` install, no network/Docker/Modal. The adapter imports
the SDK lazily inside ``main``, so the module itself must import here (where the SDK is absent).
"""

from __future__ import annotations

import dataclasses
import os

from agents_on_modal.harness import _dsh_adapter as adapter
from agents_on_modal.harness.base import HarnessSetup, McpServer, ModelAccess, SandboxProfile
from agents_on_modal.harness.dsh import (
    DEFAULT_PROFILE,
    DEFAULT_PROVIDER,
    GOAL_BLOCKED_TOKEN,
    GOAL_DONE_TOKEN,
    DeepSeek,
)
from agents_on_modal.traces import Event, envelope


def _access() -> ModelAccess:
    return ModelAccess(
        base_url="http://gw.internal:8000",
        token="wk-x.ws-y",
        model_id="deepseek-v4.1-flash-abliterated-gguf",
        context_window=32768,
        max_output_tokens=8192,
        default_effort="high",
    )


def _profile() -> SandboxProfile:
    return SandboxProfile(
        kind="modal",
        instructions="# Sandbox environment\nWork autonomously.\n",
        posix=True,
        skills={"browser": "---\nname: browser\n---\nUse agent-browser."},
        mcp_servers={"playwright": McpServer(command="mcp-server-playwright")},
        env={"PATH_EXTRA": "/opt/bin"},
    )


def _prepared() -> tuple[DeepSeek, HarnessSetup]:
    h = DeepSeek()
    setup = h.prepare(_access(), _profile(), home="/home/agent/.aom/dsh", workdir="/work")
    return h, setup


def _env(line):
    return envelope("harness", line)


# -- identity ----------------------------------------------------------------------------------
def test_identity():
    h = DeepSeek()
    assert (h.name, h.protocol, h.executable) == ("dsh", "chat", "dsh")
    assert h.pip_package == "deepseek-harness-sdk==0.2.1rc2"
    # registered so harness.get("dsh") resolves
    from agents_on_modal import harness as registry

    assert isinstance(registry.get("dsh"), DeepSeek)


# -- prepare: env ------------------------------------------------------------------------------
def test_prepare_env():
    _, setup = _prepared()
    e = setup.env
    assert e["DSH_HOME"] == "/home/agent/.aom/dsh"
    # chat-completions provider: base_url carries /v1 (provider appends /chat/completions)
    assert e["DEEPSEEK_BASE_URL"] == "http://gw.internal:8000/v1"
    assert e["AOM_DSH_BASE_URL"] == "http://gw.internal:8000/v1"
    assert e["DEEPSEEK_API_KEY"] == "wk-x.ws-y"
    assert e["AOM_DSH_API_KEY"] == "wk-x.ws-y"
    assert e["AOM_DSH_MODEL"] == "deepseek-v4.1-flash-abliterated-gguf"
    assert e["AOM_DSH_PROVIDER"] == DEFAULT_PROVIDER == "deepseek-official"
    assert e["AOM_DSH_PROFILE"] == DEFAULT_PROFILE == "sdk-minimal"
    assert e["AOM_DSH_MAX_TOKENS"] == "8192"
    assert e["AOM_DSH_WORKDIR"] == "/work"
    assert int(e["AOM_DSH_REQUEST_TIMEOUT"]) >= 3600
    assert e["AOM_DSH_INSTRUCTIONS_FILE"] == "/home/agent/.aom/dsh/instructions.md"
    # sandbox-supplied env merged
    assert e["PATH_EXTRA"] == "/opt/bin"


def test_prepare_instructions_fold_skills():
    _, setup = _prepared()
    doc = setup.files["/home/agent/.aom/dsh/instructions.md"]
    assert doc.startswith("# Sandbox environment")
    # each skill folded into the prepended instructions (dsh does not read the on-disk tree)
    assert "## Skill: browser" in doc
    assert "Use agent-browser." in doc
    # skills still written on disk for parity/inspection
    assert "/home/agent/.aom/dsh/skills/browser/SKILL.md" in setup.files


def test_prepare_windows_paths():
    h = DeepSeek()
    setup = h.prepare(
        _access(),
        SandboxProfile(kind="local", instructions="x", posix=False),
        home="C:\\home\\dsh",
        workdir="C:\\work",
    )
    assert setup.env["AOM_DSH_INSTRUCTIONS_FILE"] == "C:\\home\\dsh\\instructions.md"
    assert "C:\\home\\dsh\\instructions.md" in setup.files


# -- argv --------------------------------------------------------------------------------------
def test_interactive_argv():
    h, setup = _prepared()
    assert h.interactive_argv(setup) == ["dsh", "--profile", "sdk-minimal"]


def test_headless_argv_plain():
    h, setup = _prepared()
    argv = h.headless_argv(setup, "say hi", goal=False, resume=None)
    assert argv[:3] == ["python3", "-m", "agents_on_modal.harness._dsh_adapter"]
    assert argv[argv.index("--message") + 1] == "say hi"
    assert "--resume" not in argv and "--goal" not in argv


def test_headless_argv_goal_flag_only():
    # headless_argv passes the message verbatim; the adapter (not argv) adds the goal preamble.
    h, setup = _prepared()
    argv = h.headless_argv(setup, "ship it", goal=True, resume=None)
    assert "--goal" in argv
    assert argv[argv.index("--message") + 1] == "ship it"
    assert GOAL_DONE_TOKEN not in " ".join(argv)


def test_headless_argv_resume():
    h, setup = _prepared()
    argv = h.headless_argv(setup, "continue", goal=True, resume="sess123")
    assert argv[argv.index("--resume") + 1] == "sess123"
    assert "--goal" in argv


# -- parse -------------------------------------------------------------------------------------
def test_parse_session_and_session_id():
    h = DeepSeek()
    events = h.parse(_env({"type": "session", "session_id": "abc"}))
    assert events[0].kind == "session" and events[0].data["session_id"] == "abc"
    assert h.session_id(events) == "abc"


def test_parse_text_and_thinking():
    h = DeepSeek()
    assert h.parse(_env({"type": "text", "text": "hi"}))[0].kind == "text"
    t = h.parse(_env({"type": "thinking", "text": "mmm", "agent": "a1"}))[0]
    assert t.kind == "thinking" and t.agent == "a1"


def test_parse_tool_call_and_result():
    h = DeepSeek()
    call = h.parse(_env({"type": "tool_call", "id": "t1", "name": "shell", "input": {"command": "ls"}}))[0]
    assert call.kind == "tool_call" and call.data["name"] == "shell"
    assert call.data["input"] == {"command": "ls"}
    res = h.parse(_env({"type": "tool_result", "id": "t1", "output": "a.txt", "is_error": False}))[0]
    assert res.kind == "tool_result" and res.data["is_error"] is False


def test_parse_usage():
    h = DeepSeek()
    u = h.parse(_env({"type": "usage", "input_tokens": 100, "output_tokens": 20}))[0]
    assert u.kind == "usage" and u.data == {"input_tokens": 100, "output_tokens": 20}


def test_parse_turn_end_success_and_error():
    h = DeepSeek()
    ok = h.parse(_env({"type": "turn_end", "is_error": False, "finish_reason": "completed", "result": "done"}))
    assert [e.kind for e in ok] == ["turn_end"]
    assert ok[0].data["result"] == "done" and ok[0].data["finish_reason"] == "completed"
    bad = h.parse(_env({"type": "turn_end", "is_error": True, "finish_reason": "error", "result": "boom"}))
    assert [e.kind for e in bad] == ["turn_end", "error"]
    assert bad[1].data["message"] == "boom"


def test_parse_error_log_raw_and_nonjson():
    h = DeepSeek()
    assert h.parse(_env({"type": "error", "message": "x"}))[0].kind == "error"
    assert h.parse(_env({"type": "log", "text": "noise"}))[0].kind == "log"
    assert h.parse(_env({"type": "raw", "src": "notif", "n": {"a": 1}})) == []  # debug dump ignored
    assert h.parse(envelope("harness", "not json")) [0].kind == "log"


def test_parse_subagent():
    h = DeepSeek()
    ev = h.parse(_env({"type": "subagent", "id": "s1", "status": "completed",
                       "description": "do x", "agent": "s1"}))[0]
    assert ev.kind == "subagent" and ev.agent == "s1"
    assert ev.data["status"] == "completed" and ev.data["description"] == "do x"


# -- goal_state --------------------------------------------------------------------------------
def test_goal_state_sentinel():
    h = DeepSeek()
    s = HarnessSetup(harness="dsh", home="/h", workdir="/work", env={})
    assert h.goal_state(s, [Event("text", {"text": "verified.\nGOAL_COMPLETE"})], None) == "complete"
    assert h.goal_state(s, [Event("text", {"text": "stuck\nGOAL_BLOCKED\n"})], None) == "blocked"
    # a bare mention does not terminate the goal
    assert h.goal_state(s, [Event("text", {"text": f"I will print {GOAL_DONE_TOKEN} later"})], None) == "active"
    # last terminal signal wins
    seq = [Event("text", {"text": GOAL_BLOCKED_TOKEN}), Event("text", {"text": f"fixed\n{GOAL_DONE_TOKEN}"})]
    assert h.goal_state(s, seq, None) == "complete"
    assert h.goal_state(s, [], None) == "active"


# ============================================================================================== #
# adapter (pure helpers; no SDK)                                                                 #
# ============================================================================================== #
@dataclasses.dataclass
class _DC:
    a: int
    b: str


class _WithDump:
    def model_dump(self):
        return {"x": 1, "y": [2, 3]}


class _WithDict:
    def __init__(self):
        self.foo = "bar"
        self._hidden = "skip"


def test_to_plain_shapes():
    assert adapter.to_plain({"a": [1, "s", None]}) == {"a": [1, "s", None]}
    assert adapter.to_plain(_DC(1, "x")) == {"a": 1, "b": "x"}
    assert adapter.to_plain(_WithDump()) == {"x": 1, "y": [2, 3]}
    assert adapter.to_plain(_WithDict()) == {"foo": "bar"}  # underscore-prefixed dropped
    # depth cap falls back to str without raising
    assert isinstance(adapter.to_plain({"n": {"n": {"n": {}}}}), dict)


# Verified dsh envelope: {"method":"session.event","payload":{"sessionId","event":{"type","data"}}}
def _ev(etype, data=None):
    return {"method": "session.event",
            "payload": {"sessionId": "s1", "event": {"type": etype, "seq": 0, "data": data or {}}}}


def test_normalize_assistant_message_text_and_tool_block():
    # a committed assistant message carries text and tool-call blocks in its content
    out = adapter.normalize(_ev("assistant/message", {"message": {"role": "assistant", "content": [
        {"type": "text", "text": "hi"},
        {"type": "tool_call", "id": "c1", "name": "bash", "input": {"cmd": "ls"}},
    ]}}))
    assert out == [
        {"type": "text", "text": "hi"},
        {"type": "tool_call", "id": "c1", "name": "bash", "input": {"cmd": "ls"}},
    ]


def test_normalize_tool_call_and_result_events():
    call = adapter.normalize(_ev("tool/call", {"id": "c1", "name": "bash", "arguments": {"cmd": "ls"}}))
    assert call == [{"type": "tool_call", "id": "c1", "name": "bash", "input": {"cmd": "ls"}}]
    res = adapter.normalize(_ev("tool/result", {"callId": "c1", "output": "ok", "status": "success"}))
    assert res[0]["type"] == "tool_result" and res[0]["id"] == "c1" and res[0]["is_error"] is False
    err = adapter.normalize(_ev("tool/result", {"id": "c2", "status": "failed"}))
    assert err[0]["is_error"] is True


def test_normalize_reasoning_usage_subagent():
    assert adapter.normalize(_ev("assistant/reasoning", {"text": "think"})) == [
        {"type": "thinking", "text": "think"}]
    u = adapter.normalize(_ev("llm/usage", {"usage": {"inputTokens": 9, "outputTokens": 4}}))
    assert u == [{"type": "usage", "input_tokens": 9, "output_tokens": 4}]
    sub = adapter.normalize(_ev("agent/spawn", {"agentId": "a2", "status": "running", "prompt": "sub"}))
    assert sub[0]["type"] == "subagent" and sub[0]["id"] == "a2" and sub[0]["status"] == "started"
    assert sub[0]["agent"] == "a2"


def test_normalize_noise_status_and_unknown():
    # routing-noise event types are dropped; turn/end is authoritative from RunResult
    for noise in ("turn/start", "turn/end", "step/start", "user/message", "llm/retry",
                  "assistant/attempt"):
        assert adapter.normalize(_ev(noise, {"x": 1})) == []
    # session.status -> compact log; unknown event type -> compact log
    assert adapter.normalize({"method": "session.status",
                              "payload": {"status": "running"}})[0]["type"] == "log"
    assert adapter.normalize(_ev("mystery/thing", {}))[0] == {"type": "log", "text": "[dsh] mystery/thing"}


def test_normalize_content_as_plain_string():
    out = adapter.normalize(_ev("assistant/message", {"content": "hello world"}))
    assert out == [{"type": "text", "text": "hello world"}]


def test_compose_prompt(monkeypatch, tmp_path):
    # resume -> verbatim, no instructions
    monkeypatch.delenv("AOM_DSH_INSTRUCTIONS_FILE", raising=False)
    assert adapter.compose_prompt("go on", resume="s1", goal=True) == "go on"
    # first turn -> instructions prepended, goal preamble appended
    inst = tmp_path / "instructions.md"
    inst.write_text("# Sandbox\nrules", encoding="utf-8")
    monkeypatch.setenv("AOM_DSH_INSTRUCTIONS_FILE", str(inst))
    composed = adapter.compose_prompt("do the task", resume=None, goal=True)
    assert composed.startswith("# Sandbox\nrules")
    assert "do the task" in composed
    assert GOAL_DONE_TOKEN in composed
    # non-goal first turn -> no preamble
    plain = adapter.compose_prompt("do it", resume=None, goal=False)
    assert plain.startswith("# Sandbox\nrules") and GOAL_DONE_TOKEN not in plain


def test_supported_kwargs_filter():
    def f(a, b, c=3):
        return (a, b, c)

    assert adapter._supported(f, {"a": 1, "b": 2, "z": 9}) == {"a": 1, "b": 2}

    def g(a, **kw):
        return a

    # a callable that declares **kwargs takes everything
    assert adapter._supported(g, {"a": 1, "z": 9}) == {"a": 1, "z": 9}


def test_build_kwargs_from_env(monkeypatch):
    monkeypatch.setenv("AOM_DSH_PROVIDER", "deepseek-official")
    monkeypatch.setenv("AOM_DSH_MODEL", "m1")
    monkeypatch.setenv("DSH_HOME", "/h")
    monkeypatch.setenv("AOM_DSH_WORKDIR", "/work")
    monkeypatch.setenv("AOM_DSH_PROFILE", "sdk-minimal")
    monkeypatch.setenv("AOM_DSH_BASE_URL", "http://gw/v1")
    monkeypatch.setenv("AOM_DSH_API_KEY", "tok")
    monkeypatch.setenv("AOM_DSH_MAX_TOKENS", "8192")
    monkeypatch.setenv("AOM_DSH_REQUEST_TIMEOUT", "7200")
    monkeypatch.setenv("AOM_DSH_EFFORT", "high")
    monkeypatch.delenv("AOM_DSH_SEND_EFFORT", raising=False)
    kw = adapter._build_kwargs()
    assert kw["model"] == "m1" and kw["dsh_home"] == "/h" and kw["base_url"] == "http://gw/v1"
    assert kw["max_tokens"] == 8192 and kw["request_timeout_seconds"] == 7200.0
    # effort omitted unless AOM_DSH_SEND_EFFORT=1 (avoids a route-effort validation error)
    assert "reasoning_effort" not in kw
    monkeypatch.setenv("AOM_DSH_SEND_EFFORT", "1")
    assert adapter._build_kwargs()["reasoning_effort"] == "high"


@dataclasses.dataclass
class _FakeConfig:
    provider: str = "deepseek-official"
    model: str = "m"
    base_url: str | None = None
    api_key: str | None = None
    dsh_home: str | None = None
    cwd: str | None = None
    profile: str = "sdk-minimal"
    max_tokens: int | None = None
    request_timeout_seconds: float | None = None
    reasoning_effort: str | None = None


class _FakeModule:
    DeepSeekHarnessConfig = _FakeConfig


def test_config_field_names_and_filter():
    names = adapter._config_field_names(_FakeModule())
    assert "base_url" in names and "api_key" in names and "profile" in names
    # on_notification / on_user_question are NOT config fields (they are run() params / unsupported),
    # so the ctor filter drops them -- the real SDK forwards **kwargs to a strict dataclass.
    kw = adapter._filter_ctor_kwargs(_FakeModule(), {
        "provider": "deepseek-official", "model": "m", "base_url": "u", "api_key": "k",
        "on_notification": lambda n: None, "on_user_question": lambda *a: "", "bogus": 1,
    })
    assert kw == {"provider": "deepseek-official", "model": "m", "base_url": "u", "api_key": "k"}


def test_config_field_names_fallback_without_class():
    class _Empty:
        pass

    names = adapter._config_field_names(_Empty())
    # falls back to the documented set
    assert {"provider", "model", "base_url", "api_key", "dsh_home", "profile"} <= names


def test_gateway_root():
    assert adapter._gateway_root("http://gw:8000/v1") == "http://gw:8000"
    assert adapter._gateway_root("http://gw:8000/v1/") == "http://gw:8000"
    assert adapter._gateway_root("http://gw:8000") == "http://gw:8000"
    assert adapter._gateway_root(None) is None


def test_auth_proxy_injects_bearer_and_forwards_path():
    """The loopback proxy must replace the client's auth with Bearer <token> and forward the path,
    so dsh's rejected credential is swapped for one the Modal edge accepts."""

    import http.server
    import threading
    import urllib.request

    from agents_on_modal.harness import _dsh_auth_proxy

    seen: dict[str, object] = {}

    class Upstream(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):  # noqa: D401 - silence
            return

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            seen["body"] = self.rfile.read(n)
            seen["auth"] = self.headers.get("Authorization")
            seen["xapikey"] = self.headers.get("x-api-key")
            seen["path"] = self.path
            resp = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    up_url = f"http://127.0.0.1:{up.server_address[1]}"

    base, proxy = _dsh_auth_proxy.start(up_url, "wk-x.ws-y")
    try:
        req = urllib.request.Request(
            base + "/v1/messages", data=b'{"hi":1}',
            headers={"Content-Type": "application/json", "x-api-key": "drop-me",
                     "Authorization": "Bearer client-cred"})
        with urllib.request.urlopen(req, timeout=10) as r:
            out = r.read()
        assert out == b'{"ok":true}'
        assert seen["auth"] == "Bearer wk-x.ws-y"        # injected
        assert seen["xapikey"] is None                   # client auth stripped
        assert seen["path"] == "/v1/messages"            # path forwarded unchanged
        assert seen["body"] == b'{"hi":1}'
    finally:
        proxy.shutdown()
        up.shutdown()
