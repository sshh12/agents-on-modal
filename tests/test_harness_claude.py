"""Unit tests for the Claude Code harness (offline; no real CLI, network, Docker or Modal)."""

from __future__ import annotations

import json

from agents_on_modal.harness.base import HarnessSetup, McpServer, ModelAccess, SandboxProfile
from agents_on_modal.harness.claude import ClaudeCode
from agents_on_modal.traces import Event, envelope


def _access() -> ModelAccess:
    return ModelAccess(
        base_url="http://gw.internal:8000",
        token="wk-x.ws-y",
        model_id="glm-5.3-flash",
        context_window=131072,
        max_output_tokens=32000,
        default_effort="high",
        vision=False,
    )


def _profile() -> SandboxProfile:
    return SandboxProfile(
        kind="docker",
        instructions="# Sandbox\nWork autonomously.\n",
        posix=True,
        skills={"goal": "---\nname: goal\n---\nDo the goal."},
        mcp_servers={
            "playwright": McpServer(
                command="playwright-mcp", args=("--headless", "--isolated"), env={"X": "1"}
            )
        },
        env={"PATH_EXTRA": "/opt/bin"},
    )


def _prepared() -> tuple[ClaudeCode, HarnessSetup]:
    h = ClaudeCode()
    setup = h.prepare(_access(), _profile(), home="/home/agent/.aom/claude", workdir="/work")
    return h, setup


# -- registry / identity -----------------------------------------------------------------------
def test_identity():
    h = ClaudeCode()
    assert (h.name, h.protocol, h.executable) == ("claude", "anthropic", "claude")
    assert h.npm_package == "@anthropic-ai/claude-code@2.1.310"


# -- prepare: env ------------------------------------------------------------------------------
def test_prepare_env_points_at_gateway_single_model():
    _, setup = _prepared()
    env = setup.env
    assert env["ANTHROPIC_BASE_URL"] == "http://gw.internal:8000"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "wk-x.ws-y"
    assert "ANTHROPIC_API_KEY" not in env  # bearer, no x-api-key prompt
    # every alias + subagent pinned to the one served id
    for key in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert env[key] == "glm-5.3-flash"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
    assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "high"
    assert env["CLAUDE_CONFIG_DIR"] == "/home/agent/.aom/claude"
    assert env["PATH_EXTRA"] == "/opt/bin"  # profile env merged


def test_prepare_env_telemetry_and_sandbox_off():
    _, setup = _prepared()
    env = setup.env
    for key in (
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
        "DISABLE_AUTOUPDATER",
        "DISABLE_TELEMETRY",
        "DISABLE_ERROR_REPORTING",
        "IS_SANDBOX",
    ):
        assert env[key] in ("1", "true", "false")
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert env["ENABLE_TOOL_SEARCH"] == "false"
    assert env["IS_SANDBOX"] == "1"
    assert int(env["API_TIMEOUT_MS"]) >= 600000


def test_model_id_never_claude_prefixed():
    _, setup = _prepared()
    assert not setup.env["ANTHROPIC_MODEL"].startswith("claude-")


# -- prepare: files ----------------------------------------------------------------------------
def test_prepare_seeds_onboarding_and_trust():
    _, setup = _prepared()
    claude_json = json.loads(setup.files["/home/agent/.aom/claude/.claude.json"])
    assert claude_json["hasCompletedOnboarding"] is True
    assert claude_json["bypassPermissionsModeAccepted"] is True
    assert claude_json["projects"]["/work"]["hasTrustDialogAccepted"] is True


def test_prepare_settings_bypass_mode():
    _, setup = _prepared()
    settings = json.loads(setup.files["/home/agent/.aom/claude/settings.json"])
    assert settings["permissions"]["defaultMode"] == "bypassPermissions"
    assert settings["skipDangerousModePermissionPrompt"] is True


def test_prepare_instructions_skills_mcp_files():
    _, setup = _prepared()
    assert setup.files["/home/agent/.aom/claude/CLAUDE.md"].startswith("# Sandbox")
    assert "/home/agent/.aom/claude/skills/goal/SKILL.md" in setup.files
    mcp = json.loads(setup.files["/home/agent/.aom/claude/mcp-servers.json"])
    assert mcp["mcpServers"]["playwright"]["command"] == "playwright-mcp"
    assert mcp["mcpServers"]["playwright"]["args"] == ["--headless", "--isolated"]


def test_windows_project_key_uses_forward_slashes():
    h = ClaudeCode()
    setup = h.prepare(
        _access(),
        SandboxProfile(kind="local", instructions="x", posix=False),
        home="C:\\home\\cfg",
        workdir="C:\\Users\\a\\work",
    )
    claude_json = json.loads(setup.files["C:\\home\\cfg\\.claude.json"])
    assert "C:/Users/a/work" in claude_json["projects"]


# -- argv --------------------------------------------------------------------------------------
def test_interactive_argv():
    h, setup = _prepared()
    argv = h.interactive_argv(setup)
    assert argv[0] == "claude"
    assert "--dangerously-skip-permissions" in argv
    assert "--mcp-config" in argv and "--strict-mcp-config" in argv
    assert "-p" not in argv


def test_headless_argv_plain():
    h, setup = _prepared()
    argv = h.headless_argv(setup, "say hi", goal=False, resume=None)
    assert argv[:3] == ["claude", "-p", "say hi"]
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv and "--dangerously-skip-permissions" in argv
    assert "--resume" not in argv


def test_headless_argv_goal_wraps_slash_goal():
    h, setup = _prepared()
    argv = h.headless_argv(setup, "make done.txt exist", goal=True, resume=None)
    assert argv[2] == "/goal make done.txt exist"


def test_headless_argv_resume_uses_prompt_verbatim():
    h, setup = _prepared()
    cont = h.continuation_prompt(goal=True)
    argv = h.headless_argv(setup, cont, goal=True, resume="sess-123")
    assert argv[2] == cont  # resume: prompt is the continuation message, not re-wrapped
    assert argv[-2:] == ["--resume", "sess-123"]


# -- parse -------------------------------------------------------------------------------------
def _env(line):
    return envelope("harness", line)


def test_parse_system_init_is_session():
    h = ClaudeCode()
    events = h.parse(
        _env({"type": "system", "subtype": "init", "session_id": "abc", "model": "glm", "cwd": "/work", "tools": ["Bash"]})
    )
    assert len(events) == 1 and events[0].kind == "session"
    assert events[0].data["session_id"] == "abc"
    assert h.session_id(events) == "abc"


def test_parse_assistant_text_thinking_tool():
    h = ClaudeCode()
    msg = {
        "type": "assistant",
        "message": {
            "id": "m1",
            "content": [
                {"type": "thinking", "thinking": "let me think"},
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
            ],
            "usage": {"input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 10},
        },
    }
    events = h.parse(_env(msg))
    kinds = [e.kind for e in events]
    assert kinds == ["thinking", "text", "tool_call", "usage"]
    assert events[1].data["text"] == "hello"
    assert events[2].data["name"] == "Bash" and events[2].data["input"] == {"command": "ls"}
    assert events[3].data["context_tokens"] == 110


def test_parse_user_tool_result():
    h = ClaudeCode()
    msg = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"tool_use_id": "toolu_1", "type": "tool_result", "content": "ok", "is_error": False}
            ],
        },
    }
    events = h.parse(_env(msg))
    assert len(events) == 1 and events[0].kind == "tool_result"
    assert events[0].data == {"id": "toolu_1", "output": "ok", "is_error": False}


def test_parse_compact_boundary():
    h = ClaudeCode()
    events = h.parse(
        _env(
            {
                "type": "system",
                "subtype": "compact_boundary",
                "compact_metadata": {"trigger": "auto", "pre_tokens": 190057, "post_tokens": 2569},
            }
        )
    )
    assert events[0].kind == "compaction"
    assert events[0].data == {"pre_tokens": 190057, "post_tokens": 2569}


def test_parse_result_success():
    h = ClaudeCode()
    events = h.parse(
        _env(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "done",
                "session_id": "abc",
                "num_turns": 3,
                "usage": {"input_tokens": 500, "output_tokens": 80},
            }
        )
    )
    kinds = [e.kind for e in events]
    assert "session" in kinds and "usage" in kinds and "turn_end" in kinds
    turn = next(e for e in events if e.kind == "turn_end")
    assert turn.data["subtype"] == "success" and turn.data["is_error"] is False


def test_parse_result_error_emits_error():
    h = ClaudeCode()
    events = h.parse(
        _env({"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "API Error: 503"})
    )
    assert any(e.kind == "error" for e in events)
    turn = next(e for e in events if e.kind == "turn_end")
    assert turn.data["is_error"] is True


def test_parse_non_json_line_is_log():
    h = ClaudeCode()
    events = h.parse(envelope("harness", "plain stderr noise"))
    assert len(events) == 1 and events[0].kind == "log"
    assert events[0].data["text"] == "plain stderr noise"


# -- goal_state --------------------------------------------------------------------------------
def test_goal_state_complete_on_success():
    h = ClaudeCode()
    events = [Event("turn_end", {"subtype": "success", "is_error": False})]
    assert h.goal_state(_setup(), events, None) == "complete"


def test_goal_state_active_on_error_or_missing():
    h = ClaudeCode()
    assert h.goal_state(_setup(), [], None) == "active"
    err = [Event("turn_end", {"subtype": "error_max_turns", "is_error": True})]
    assert h.goal_state(_setup(), err, None) == "active"


def _setup() -> HarnessSetup:
    return HarnessSetup(harness="claude", home="/h", workdir="/work", env={})
