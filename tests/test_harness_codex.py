"""Unit tests for the Codex harness (offline; no real CLI, network, Docker or Modal)."""

from __future__ import annotations

import json

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback on 3.10
    tomllib = None

from agents_on_modal.harness.base import HarnessSetup, McpServer, ModelAccess, SandboxProfile
from agents_on_modal.harness.codex import Codex
from agents_on_modal.traces import Event, envelope


def _access(vision: bool = False) -> ModelAccess:
    return ModelAccess(
        base_url="http://gw.internal:8000",
        token="wk-x.ws-y",
        model_id="glm-5.3-flash",
        context_window=131072,
        max_output_tokens=32000,
        default_effort="high",
        vision=vision,
    )


def _profile() -> SandboxProfile:
    return SandboxProfile(
        kind="modal",
        instructions="# Sandbox\nWork autonomously.\n",
        posix=True,
        skills={"goal": "---\nname: goal\n---\nDo the goal."},
        mcp_servers={
            "playwright": McpServer(
                command="playwright-mcp", args=("--headless", "--isolated"), env={"OUT": "/tmp/pw"}
            )
        },
        env={"PATH_EXTRA": "/opt/bin"},
    )


def _prepared(vision: bool = False) -> tuple[Codex, HarnessSetup]:
    h = Codex()
    setup = h.prepare(_access(vision), _profile(), home="/home/agent/.aom/codex", workdir="/work")
    return h, setup


def _config(setup: HarnessSetup) -> str:
    return setup.files["/home/agent/.aom/codex/config.toml"]


# -- identity ----------------------------------------------------------------------------------
def test_identity():
    h = Codex()
    assert (h.name, h.protocol, h.executable) == ("codex", "responses", "codex")
    assert h.npm_package == "@openai/codex@0.155.1"


# -- prepare: env --------------------------------------------------------------------------------
def test_prepare_env():
    _, setup = _prepared()
    assert setup.env["CODEX_HOME"] == "/home/agent/.aom/codex"
    assert setup.env["AOM_TOKEN"] == "wk-x.ws-y"
    assert setup.env["PATH_EXTRA"] == "/opt/bin"


# -- prepare: config.toml ----------------------------------------------------------------------
def test_config_toml_values():
    _, setup = _prepared()
    toml = _config(setup)
    if tomllib is not None:
        cfg = tomllib.loads(toml)
        assert cfg["model"] == "glm-5.3-flash"
        assert cfg["model_provider"] == "gw"
        assert cfg["model_reasoning_summary"] == "none"
        assert cfg["web_search"] == "disabled"
        assert cfg["approval_policy"] == "never"
        assert cfg["sandbox_mode"] == "danger-full-access"
        assert cfg["check_for_update_on_startup"] is False
        prov = cfg["model_providers"]["gw"]
        assert prov["name"] != "OpenAI"
        assert prov["base_url"] == "http://gw.internal:8000/v1"
        assert prov["wire_api"] == "responses"
        assert prov["env_key"] == "AOM_TOKEN"
        assert prov["requires_openai_auth"] is False
        assert prov["supports_websockets"] is False
        assert prov["stream_idle_timeout_ms"] == 1800000
        # CRITICAL: a dead endpoint must not retry connection failures forever.
        assert cfg["features"]["unbounded_connection_retries"] is False
        assert cfg["analytics"]["enabled"] is False
        assert cfg["otel"]["exporter"] == "none"
        assert cfg["projects"]["/work"]["trust_level"] == "trusted"
        # MCP server registered in-config.
        assert cfg["mcp_servers"]["playwright"]["command"] == "playwright-mcp"
        assert cfg["mcp_servers"]["playwright"]["args"] == ["--headless", "--isolated"]
        assert cfg["mcp_servers"]["playwright"]["env"]["OUT"] == "/tmp/pw"
    else:  # pragma: no cover
        assert 'model_provider = "gw"' in toml
        assert "unbounded_connection_retries = false" in toml


def test_config_catalog_path_points_into_home():
    _, setup = _prepared()
    toml = _config(setup)
    assert '/home/agent/.aom/codex/catalog.json' in toml


# -- prepare: catalog.json ---------------------------------------------------------------------
def test_catalog_single_minimal_model():
    _, setup = _prepared()
    catalog = json.loads(setup.files["/home/agent/.aom/codex/catalog.json"])
    assert len(catalog["models"]) == 1
    model = catalog["models"][0]
    assert model["slug"] == "glm-5.3-flash"
    # fields required to parse (research §2)
    for field in (
        "display_name",
        "supported_reasoning_levels",
        "shell_type",
        "visibility",
        "supported_in_api",
        "priority",
        "support_verbosity",
        "truncation_policy",
        "experimental_supported_tools",
        "base_instructions",
    ):
        assert field in model
    assert model["shell_type"] == "shell_command"
    assert model["apply_patch_tool_type"] == "freeform"  # gateway maps the custom tool
    assert model["default_reasoning_summary"] == "none"
    assert model["context_window"] == 131072
    assert model["max_context_window"] == 131072
    assert model["input_modalities"] == ["text"]
    assert model["default_reasoning_level"] == "high"


def test_catalog_vision_modalities():
    _, setup = _prepared(vision=True)
    catalog = json.loads(setup.files["/home/agent/.aom/codex/catalog.json"])
    assert catalog["models"][0]["input_modalities"] == ["text", "image"]


def test_prepare_instructions_and_skills():
    _, setup = _prepared()
    assert setup.files["/home/agent/.aom/codex/AGENTS.md"].startswith("# Sandbox")
    assert "/home/agent/.aom/codex/skills/goal/SKILL.md" in setup.files


def test_windows_trust_key_quoted():
    h = Codex()
    setup = h.prepare(
        _access(),
        SandboxProfile(kind="local", instructions="x", posix=False),
        home="C:\\home\\codex",
        workdir="C:\\Users\\a\\work",
    )
    toml = setup.files["C:\\home\\codex\\config.toml"]
    # backslashes escaped inside the TOML basic string
    assert "C:\\\\Users\\\\a\\\\work" in toml


# -- argv --------------------------------------------------------------------------------------
def test_interactive_argv():
    h, setup = _prepared()
    assert h.interactive_argv(setup) == ["codex", "--dangerously-bypass-approvals-and-sandbox"]


def test_headless_argv_plain():
    h, setup = _prepared()
    argv = h.headless_argv(setup, "say hi", goal=False, resume=None)
    assert argv[:2] == ["codex", "exec"]
    assert "--json" in argv and "--skip-git-repo-check" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert argv[-1] == "say hi"
    assert "resume" not in argv


def test_headless_argv_goal_adds_preamble():
    h, setup = _prepared()
    argv = h.headless_argv(setup, "ship the feature", goal=True, resume=None)
    assert argv[-1].startswith("ship the feature")
    # Goal termination is signalled by a textual sentinel (tool-independent), not update_goal.
    assert "GOAL_COMPLETE" in argv[-1]


def test_headless_argv_resume_form():
    h, setup = _prepared()
    cont = h.continuation_prompt(goal=True)
    argv = h.headless_argv(setup, cont, goal=True, resume="01a172d1")
    assert argv[:4] == ["codex", "exec", "resume", "01a172d1"]
    assert argv[-1] == cont
    assert "GOAL_COMPLETE" in cont


# -- parse -------------------------------------------------------------------------------------
def _env(line):
    return envelope("harness", line)


def test_parse_thread_started_session():
    h = Codex()
    events = h.parse(_env({"type": "thread.started", "thread_id": "01a172d1"}))
    assert events[0].kind == "session" and events[0].data["session_id"] == "01a172d1"
    assert h.session_id(events) == "01a172d1"


def test_parse_agent_message_and_reasoning():
    h = Codex()
    msg = h.parse(_env({"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}))[0]
    assert msg.kind == "text" and msg.data == {"text": "hi"}
    assert h.parse(_env({"type": "item.completed", "item": {"type": "reasoning", "text": "mmm"}}))[0].kind == "thinking"


def test_parse_command_execution_call_and_result():
    h = Codex()
    item = {
        "id": "item_0",
        "type": "command_execution",
        "command": "echo hi",
        "aggregated_output": "hi\n",
        "exit_code": 0,
        "status": "completed",
    }
    events = h.parse(_env({"type": "item.completed", "item": item}))
    assert [e.kind for e in events] == ["tool_call", "tool_result"]
    assert events[0].data["input"] == {"command": "echo hi"}
    assert events[1].data["output"] == "hi\n" and events[1].data["is_error"] is False


def test_parse_command_execution_nonzero_is_error():
    h = Codex()
    item = {"id": "i", "type": "command_execution", "command": "false", "aggregated_output": "", "exit_code": 1}
    events = h.parse(_env({"type": "item.completed", "item": item}))
    assert events[1].data["is_error"] is True


def test_parse_file_change_is_tool_result():
    h = Codex()
    item = {"id": "i1", "type": "file_change", "changes": [{"path": "hello.txt", "kind": "add"}], "status": "completed"}
    events = h.parse(_env({"type": "item.completed", "item": item}))
    assert len(events) == 1 and events[0].kind == "tool_result"
    assert events[0].data["is_error"] is False


def test_parse_collab_tool_call_is_subagent():
    h = Codex()
    item = {
        "type": "collab_tool_call",
        "tool": "spawn_agent",
        "prompt": "do subtask",
        "agents_states": {"01a1": {"status": "pending_init", "message": None}},
        "status": "completed",
    }
    events = h.parse(_env({"type": "item.completed", "item": item}))
    assert events[0].kind == "subagent" and events[0].agent == "01a1"
    assert events[0].data["status"] == "started"


def test_parse_turn_completed_usage_and_turn_end():
    h = Codex()
    events = h.parse(
        _env(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 440, "output_tokens": 32, "cached_input_tokens": 0},
            }
        )
    )
    usage = next(e for e in events if e.kind == "usage")
    assert usage.data["input_tokens"] == 440 and usage.data["output_tokens"] == 32
    assert any(e.kind == "turn_end" and e.data["is_error"] is False for e in events)


def test_parse_turn_failed_and_error():
    h = Codex()
    events = h.parse(_env({"type": "turn.failed", "error": {"message": "Connection failed"}}))
    assert any(e.kind == "turn_end" and e.data["is_error"] is True for e in events)
    assert any(e.kind == "error" and e.data["message"] == "Connection failed" for e in events)
    err = h.parse(_env({"type": "error", "message": "Reconnecting..."}))
    assert err[0].kind == "error"


def test_parse_item_error():
    h = Codex()
    events = h.parse(_env({"type": "item.completed", "item": {"type": "error", "message": "boom"}}))
    assert events[0].kind == "error" and events[0].data["message"] == "boom"


def test_parse_non_json_is_log():
    h = Codex()
    events = h.parse(envelope("harness", "warning: noise"))
    assert events[0].kind == "log"


# -- goal parsing + goal_state -----------------------------------------------------------------
def test_parse_create_goal_event():
    h = Codex()
    item = {"type": "function_call", "name": "create_goal", "arguments": json.dumps({"objective": "do x"})}
    events = h.parse(_env({"type": "item.completed", "item": item}))
    assert events[0].kind == "goal" and events[0].data["status"] == "active"
    assert events[0].data["objective"] == "do x"


def test_parse_update_goal_complete():
    h = Codex()
    item = {"type": "function_call", "name": "update_goal", "arguments": json.dumps({"status": "complete"})}
    events = h.parse(_env({"type": "item.completed", "item": item}))
    assert events[0].kind == "goal" and events[0].data["status"] == "complete"


def test_goal_state_complete_blocked_active():
    h = Codex()
    s = HarnessSetup(harness="codex", home="/h", workdir="/work", env={})
    assert h.goal_state(s, [Event("goal", {"status": "complete"})], None) == "complete"
    assert h.goal_state(s, [Event("goal", {"status": "blocked"})], None) == "blocked"
    # last goal event wins
    seq = [Event("goal", {"status": "active"}), Event("goal", {"status": "complete"})]
    assert h.goal_state(s, seq, None) == "complete"
    # create-only / none -> keep resuming
    assert h.goal_state(s, [Event("goal", {"status": "active"})], None) == "active"
    assert h.goal_state(s, [], None) == "active"


def test_goal_state_textual_sentinel():
    """The model signals termination with a standalone sentinel line (update_goal fallback aside)."""

    h = Codex()
    s = HarnessSetup(harness="codex", home="/h", workdir="/work", env={})
    done = [Event("text", {"text": "all requirements verified.\nGOAL_COMPLETE"})]
    assert h.goal_state(s, done, None) == "complete"
    blocked = [Event("text", {"text": "cannot proceed\nGOAL_BLOCKED\n"})]
    assert h.goal_state(s, blocked, None) == "blocked"
    # merely mentioning the token inline must NOT terminate the goal
    mention = [Event("text", {"text": "I will print GOAL_COMPLETE when the task is done."})]
    assert h.goal_state(s, mention, None) == "active"
    # the last terminal signal wins across text and goal events
    seq = [Event("text", {"text": "GOAL_BLOCKED"}), Event("text", {"text": "fixed it\nGOAL_COMPLETE"})]
    assert h.goal_state(s, seq, None) == "complete"
