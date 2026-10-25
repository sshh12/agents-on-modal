"""Codex CLI harness.

Turns a :class:`ModelAccess` into an isolated, single-model, telemetry-off ``CODEX_HOME`` (with a
provider that is deliberately never named ``OpenAI``) and parses ``codex exec --json`` events into
normalized :class:`Event`\\ s. Every wire detail here is from the verified research in
``research/codex.md`` (Codex CLI 0.155.1).

Codex ``/goal`` does not auto-continue under ``codex exec`` (verified), so goal mode is driven by the
in-sandbox runner: it resumes the thread (``codex exec resume``) with :meth:`continuation_prompt`
until :meth:`goal_state` sees the model call ``update_goal`` with a terminal status.
"""

from __future__ import annotations

import json
import ntpath
import posixpath
from typing import Any

from ..traces import Event
from .base import GoalState, Harness, HarnessSetup, ModelAccess, SandboxProfile

#: Normalized effort scale advertised in the single-model catalog (gateway maps each to the engine).
_REASONING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: Core agent prompt sent as the Responses ``instructions`` field. Codex's full bundled prompt is
#: not shipped with a custom ``model_catalog_json``; this is a concise functional replacement. The
#: sandbox's own instructions arrive separately via ``AGENTS.md``.
_BASE_INSTRUCTIONS = (
    "You are Codex, an autonomous software-engineering agent running in a sandboxed workspace.\n"
    "All approvals and filesystem sandboxing are disabled; the surrounding sandbox is the only "
    "security boundary, so act without asking for confirmation.\n"
    "Use the shell tool to inspect and modify the workspace and the apply_patch tool to edit files. "
    "Prefer small, verifiable steps: make a change, then run the build or tests to confirm it.\n"
    "Before claiming a task is done, verify it by running the relevant command and reading the "
    "output. Be concise in your final message and report what you did and how you verified it."
)

#: Sentinel tokens the model emits to signal goal termination. ``codex exec`` does not reliably
#: expose the create_goal/update_goal tools to an arbitrary open-weight model, so completion is
#: signalled textually (tool-independent) and detected in :meth:`Codex.goal_state`; the update_goal
#: path is kept as a fallback for when those tools are present.
GOAL_DONE_TOKEN = "GOAL_COMPLETE"
GOAL_BLOCKED_TOKEN = "GOAL_BLOCKED"

#: Appended to the first prompt in goal mode. The model works across turns and marks termination with
#: a sentinel line so the runner knows when to stop resuming (an unmet/impossible goal just keeps
#: working until the deadline, which is the intended behaviour).
_GOAL_PREAMBLE = (
    "Treat this as a long-horizon goal: keep working across turns until every requirement is met "
    "and verified. When -- and only when -- the goal is fully achieved and you have verified it "
    "with a requirement-by-requirement audit, end your final message with a line containing "
    f"exactly {GOAL_DONE_TOKEN} and nothing else on that line. If you become permanently blocked "
    f"and cannot make progress, end with a line containing exactly {GOAL_BLOCKED_TOKEN}."
)


class Codex(Harness):
    """Codex CLI (``codex``) talking the OpenAI Responses protocol to the gateway."""

    name = "codex"
    protocol = "responses"
    executable = "codex"
    npm_package = "@openai/codex@0.155.1"

    #: Function tools that manage goal state (research §8).
    _goal_tools = ("create_goal", "update_goal", "get_goal")

    # -- preparation --------------------------------------------------------------------------
    def prepare(self, access: ModelAccess, profile: SandboxProfile, *, home: str, workdir: str) -> HarnessSetup:
        join = _joiner(profile.posix)
        catalog_path = join(home, "catalog.json")

        env = {"CODEX_HOME": home, "AOM_TOKEN": access.token}
        env.update(profile.env)

        files: dict[str, str] = {
            # config.toml (CODEX_HOME must already exist; writing here creates it).
            join(home, "config.toml"): self._config_toml(access, workdir, catalog_path, profile),
            # Single-model catalog: only our slug, so subagents and the picker never leave it.
            catalog_path: self._catalog_json(access),
            # Root instructions -> global AGENTS.md.
            join(home, "AGENTS.md"): profile.instructions,
        }
        # Skills -> $CODEX_HOME/skills/<name>/SKILL.md.
        for skill_name, skill_text in profile.skills.items():
            files[join(home, "skills", skill_name, "SKILL.md")] = skill_text

        return HarnessSetup(
            harness=self.name, home=home, workdir=workdir, env=env, files=files, posix=profile.posix
        )

    def _config_toml(self, access: ModelAccess, workdir: str, catalog_path: str, profile: SandboxProfile) -> str:
        base_url = access.base_url.rstrip("/") + "/v1"  # Codex appends /responses
        lines: list[str] = [
            f"model = {_ts(access.model_id)}",
            'model_provider = "gw"',
            f"model_catalog_json = {_ts(catalog_path)}",
            f"model_reasoning_effort = {_ts(access.default_effort)}",
            'model_reasoning_summary = "none"',
            'web_search = "disabled"',
            'approval_policy = "never"',
            'sandbox_mode = "danger-full-access"',
            "check_for_update_on_startup = false",
            "",
            "[features]",
            # CRITICAL: without this a dead endpoint retries connection failures forever (research §4).
            "unbounded_connection_retries = false",
            "plugins = false",
            "apps = false",
            "remote_plugin = false",
            "tool_suggest = false",
            "",
            "[analytics]",
            "enabled = false",
            "",
            "[feedback]",
            "enabled = false",
            "",
            "[otel]",
            'exporter = "none"',
            'metrics_exporter = "none"',
            'trace_exporter = "none"',
            "",
            "[model_providers.gw]",
            'name = "gw"',  # never "OpenAI": that flips Codex into OpenAI-only behaviour
            f"base_url = {_ts(base_url)}",
            'wire_api = "responses"',
            'env_key = "AOM_TOKEN"',
            "requires_openai_auth = false",
            "supports_websockets = false",
            "stream_idle_timeout_ms = 1800000",
            "request_max_retries = 4",
            "stream_max_retries = 5",
            "",
            # Pre-trust the workspace so the TUI never shows the trust prompt.
            f"[projects.{_ts(workdir)}]",
            'trust_level = "trusted"',
        ]
        for server_name, server in profile.mcp_servers.items():
            lines += ["", f"[mcp_servers.{_bare_key(server_name)}]"]
            lines.append(f"command = {_ts(server.command)}")
            lines.append(f"args = {_ts_array(server.args)}")
            for key, value in server.env.items():
                lines.append(f"env.{_bare_key(key)} = {_ts(value)}")
            lines.append("startup_timeout_sec = 30")
            lines.append("tool_timeout_sec = 120")
        return "\n".join(lines) + "\n"

    def _catalog_json(self, access: ModelAccess) -> str:
        """One minimal valid ``ModelInfo`` (research §2). ``apply_patch_tool_type='freeform'`` is set
        because the gateway maps the custom apply_patch tool back for the backend."""

        model = access.model_id
        info = {
            "slug": model,
            "display_name": model,
            "supported_reasoning_levels": [{"effort": e, "description": e} for e in _REASONING_LEVELS],
            "default_reasoning_level": access.default_effort,
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": True,
            "priority": 0,
            "support_verbosity": False,
            "truncation_policy": {"mode": "tokens", "limit": 10000},
            "experimental_supported_tools": [],
            "base_instructions": _BASE_INSTRUCTIONS,
            "context_window": access.context_window,
            "max_context_window": access.context_window,
            "apply_patch_tool_type": "freeform",
            "default_reasoning_summary": "none",
            "input_modalities": ["text", "image"] if access.vision else ["text"],
        }
        return json.dumps({"models": [info]}, indent=2) + "\n"

    # -- argv ---------------------------------------------------------------------------------
    def interactive_argv(self, setup: HarnessSetup) -> list[str]:
        return [self.executable, "--dangerously-bypass-approvals-and-sandbox"]

    def headless_argv(self, setup: HarnessSetup, prompt: str, *, goal: bool, resume: str | None) -> list[str]:
        flags = ["--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]
        if resume:
            # resume injects the goal continuation message ahead of this prompt; one turn per call.
            return [self.executable, "exec", "resume", resume, *flags, prompt]
        text = f"{prompt}\n\n{_GOAL_PREAMBLE}" if goal else prompt
        return [self.executable, "exec", *flags, text]

    def continuation_prompt(self, *, goal: bool) -> str:
        if goal:
            return (
                "Continue working toward the goal. Keep going across turns until every requirement "
                "is met and verified. When it is fully met and verified, end your message with a "
                f"line containing exactly {GOAL_DONE_TOKEN}. If you are permanently blocked, end "
                f"with a line containing exactly {GOAL_BLOCKED_TOKEN}. Otherwise keep working."
            )
        return "Continue."

    # -- parsing ------------------------------------------------------------------------------
    def parse(self, envelope: dict[str, Any]) -> list[Event]:
        ts = float(envelope.get("ts", 0.0))
        data = envelope.get("data")
        if not isinstance(data, dict):
            text = envelope.get("text")
            return [Event("log", {"text": str(text)}, ts=ts)] if text else []

        etype = data.get("type")
        if etype == "thread.started":
            return [Event("session", {"session_id": data.get("thread_id")}, ts=ts)]
        if etype in ("item.completed", "item.updated"):
            return self._parse_item(data.get("item") or {}, ts)
        if etype == "turn.completed":
            return self._parse_turn_completed(data, ts)
        if etype == "turn.failed":
            message = (data.get("error") or {}).get("message")
            return [
                Event("turn_end", {"is_error": True, "result": message}, ts=ts),
                Event("error", {"message": message}, ts=ts),
            ]
        if etype == "error":
            return [Event("error", {"message": data.get("message")}, ts=ts)]
        # thread.started handled; turn.started / item.started carry no normalized payload.
        return []

    def _parse_item(self, item: dict[str, Any], ts: float) -> list[Event]:
        itype = item.get("type")

        # Goal management surfaces as create_goal / update_goal / get_goal function calls.
        goal_event = self._goal_event(item, ts)
        if goal_event is not None:
            return [goal_event]

        if itype == "agent_message":
            return [Event("text", {"text": item.get("text", "")}, ts=ts)]
        if itype == "reasoning":
            return [Event("thinking", {"text": item.get("text", "")}, ts=ts)]
        if itype == "command_execution":
            exit_code = item.get("exit_code")
            return [
                Event(
                    "tool_call",
                    {"id": item.get("id"), "name": "exec_command", "input": {"command": item.get("command")}},
                    ts=ts,
                ),
                Event(
                    "tool_result",
                    {
                        "id": item.get("id"),
                        "output": item.get("aggregated_output"),
                        "is_error": bool(exit_code) if exit_code is not None else False,
                    },
                    ts=ts,
                ),
            ]
        if itype == "file_change":
            return [
                Event(
                    "tool_result",
                    {"id": item.get("id"), "output": item.get("changes"), "is_error": False},
                    ts=ts,
                )
            ]
        if itype == "mcp_tool_call":
            return [
                Event(
                    "tool_call",
                    {
                        "id": item.get("id"),
                        "name": f"{item.get('server')}__{item.get('tool')}",
                        "input": item.get("arguments") or {},
                    },
                    ts=ts,
                ),
                Event(
                    "tool_result",
                    {
                        "id": item.get("id"),
                        "output": item.get("result"),
                        "is_error": item.get("status") == "failed" or bool(item.get("error")),
                    },
                    ts=ts,
                ),
            ]
        if itype == "collab_tool_call":
            events: list[Event] = []
            states = item.get("agents_states") or {}
            for agent_id, state in states.items():
                events.append(
                    Event(
                        "subagent",
                        {
                            "id": agent_id,
                            "status": _subagent_status(state.get("status")),
                            "description": item.get("prompt"),
                        },
                        agent=str(agent_id),
                        ts=ts,
                    )
                )
            if not events:  # no per-agent states yet; still record the call
                events.append(Event("subagent", {"id": None, "status": "started"}, ts=ts))
            return events
        if itype == "error":
            return [Event("error", {"message": item.get("message")}, ts=ts)]
        return []

    def _goal_event(self, item: dict[str, Any], ts: float) -> Event | None:
        """Detect a goal tool call in an exec item and map it to a normalized ``goal`` event."""

        name = item.get("name") or item.get("tool")
        if name not in self._goal_tools:
            return None
        args = _as_dict(item.get("arguments") if "arguments" in item else item.get("input"))
        if name == "update_goal":
            return Event("goal", {"status": _goal_status(args.get("status"))}, ts=ts)
        if name == "create_goal":
            payload = {"status": "active"}
            if args.get("objective"):
                payload["objective"] = args["objective"]
            return Event("goal", payload, ts=ts)
        return Event("goal", {"status": "active"}, ts=ts)  # get_goal: a goal exists

    def _parse_turn_completed(self, data: dict[str, Any], ts: float) -> list[Event]:
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return [
            Event("usage", {"input_tokens": input_tokens, "output_tokens": output_tokens}, ts=ts),
            Event("turn_end", {"is_error": False}, ts=ts),
        ]

    # -- goal ---------------------------------------------------------------------------------
    def goal_state(self, setup: HarnessSetup, events: list[Event], session_id: str | None) -> GoalState:
        """Codex exec does not auto-continue a goal, so the runner resumes until the model marks
        the goal terminal. Termination is signalled either by a sentinel line in the model's own
        message (:data:`GOAL_DONE_TOKEN` / :data:`GOAL_BLOCKED_TOKEN`) or, when available, by an
        ``update_goal`` call surfaced as a ``goal`` event. The last terminal signal in order wins;
        with none, keep resuming (an unmet/impossible goal runs until the runner's deadline).

        The sentinel must stand alone on its line so that merely *mentioning* the token (e.g. "I
        will print GOAL_COMPLETE when done") does not end the goal prematurely.
        """

        last_status: str | None = None
        for event in events:
            if event.kind == "goal" and event.data.get("status"):
                last_status = str(event.data["status"])
            elif event.kind == "text":
                for line in str(event.data.get("text") or "").splitlines():
                    stripped = line.strip()
                    if stripped == GOAL_DONE_TOKEN:
                        last_status = "complete"
                    elif stripped == GOAL_BLOCKED_TOKEN:
                        last_status = "blocked"
        if last_status == "complete":
            return "complete"
        if last_status == "blocked":
            return "blocked"
        return "active"


# -- module helpers ---------------------------------------------------------------------------
def _joiner(posix: bool):
    return (posixpath if posix else ntpath).join


def _ts(value: str) -> str:
    """A TOML basic string (double-quoted, escaped)."""

    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _ts_array(values: Any) -> str:
    return "[" + ", ".join(_ts(v) for v in values) + "]"


def _bare_key(key: str) -> str:
    """A TOML key: bare when it is a simple identifier, else quoted."""

    if key and all(c.isalnum() or c in "-_" for c in key):
        return key
    return _ts(key)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _goal_status(status: Any) -> str:
    """Map a Codex goal status onto the normalized ``goal`` event vocabulary."""

    mapping = {
        "complete": "complete",
        "blocked": "blocked",
        "paused": "blocked",
        "budgetLimited": "budget_limited",
        "usageLimited": "budget_limited",
        "active": "active",
    }
    return mapping.get(str(status), "active")


def _subagent_status(status: Any) -> str:
    text = str(status or "")
    if text in ("completed", "complete", "done"):
        return "completed"
    if text in ("failed", "error"):
        return "failed"
    return "started"
