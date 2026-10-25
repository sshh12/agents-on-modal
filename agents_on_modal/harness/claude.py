"""Claude Code harness.

Turns a :class:`ModelAccess` into a maximally permissive, single-model, isolated, telemetry-off
Claude Code home (``CLAUDE_CONFIG_DIR``) and parses its ``--output-format stream-json`` stream into
normalized :class:`Event`\\ s. Every wire detail here is from the verified research in
``research/claude-code.md`` (Claude Code 2.1.310).
"""

from __future__ import annotations

import json
import ntpath
import posixpath
from typing import Any

from ..traces import Event
from .base import GoalState, Harness, HarnessSetup, ModelAccess, SandboxProfile


class ClaudeCode(Harness):
    """Claude Code (``claude``) talking the Anthropic Messages protocol to the gateway."""

    name = "claude"
    protocol = "anthropic"
    executable = "claude"
    npm_package = "@anthropic-ai/claude-code@2.1.310"

    #: Version used to satisfy the onboarding seed (see research §5).
    _onboarding_version = "2.1.310"

    # -- preparation --------------------------------------------------------------------------
    def prepare(self, access: ModelAccess, profile: SandboxProfile, *, home: str, workdir: str) -> HarnessSetup:
        join = _joiner(profile.posix)

        env = self._env(access, home)
        env.update(profile.env)  # sandbox-supplied extras win (e.g. PATH tweaks)

        files: dict[str, str] = {}

        # First-run dialog seeds (onboarding, trust, bypass warning). The projects key is the
        # absolute workdir with forward slashes (research §5).
        files[join(home, ".claude.json")] = _json(
            {
                "hasCompletedOnboarding": True,
                "lastOnboardingVersion": self._onboarding_version,
                "bypassPermissionsModeAccepted": True,
                "customApiKeyResponses": {"approved": [], "rejected": []},
                "projects": {workdir.replace("\\", "/"): {"hasTrustDialogAccepted": True}},
            }
        )
        files[join(home, "settings.json")] = _json(
            {
                "theme": "dark",
                "skipDangerousModePermissionPrompt": True,
                "permissions": {"defaultMode": "bypassPermissions"},
            }
        )

        # Root instructions -> user-level CLAUDE.md.
        files[join(home, "CLAUDE.md")] = profile.instructions

        # Skills -> $CLAUDE_CONFIG_DIR/skills/<name>/SKILL.md.
        for skill_name, skill_text in profile.skills.items():
            files[join(home, "skills", skill_name, "SKILL.md")] = skill_text

        # MCP: a single strict config file. Always written (even empty) so --strict-mcp-config
        # locks Claude to exactly these servers and the argv is deterministic without setup.files.
        files[self._mcp_path(home, profile.posix)] = _json(
            {
                "mcpServers": {
                    name: {"command": s.command, "args": list(s.args), "env": dict(s.env)}
                    for name, s in profile.mcp_servers.items()
                }
            }
        )

        return HarnessSetup(
            harness=self.name, home=home, workdir=workdir, env=env, files=files, posix=profile.posix
        )

    def _env(self, access: ModelAccess, home: str) -> dict[str, str]:
        """Complete env for a single-model, isolated, telemetry-off run (research §§1,3,4,9)."""

        model = access.model_id
        return {
            "CLAUDE_CONFIG_DIR": home,
            # Point at the gateway; Bearer token, no prompt.
            "ANTHROPIC_BASE_URL": access.base_url,
            "ANTHROPIC_AUTH_TOKEN": access.token,
            # Pin every alias to the one served model so nothing resolves to a real claude-* id.
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            # Declare the real window/output so auto-compaction triggers correctly.
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(access.context_window),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(access.max_output_tokens),
            # Effort default from the catalog (gateway maps the level).
            "CLAUDE_CODE_EFFORT_LEVEL": access.default_effort,
            # No traffic that bypasses the gateway.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "ENABLE_TOOL_SEARCH": "false",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            # Allow running bypass mode as root inside a container.
            "IS_SANDBOX": "1",
            # Long header wait for a cold-starting, autoscaling backend.
            "API_TIMEOUT_MS": "3600000",
        }

    def _mcp_path(self, home: str, posix: bool) -> str:
        return _joiner(posix)(home, "mcp-servers.json")

    # -- argv ---------------------------------------------------------------------------------
    def interactive_argv(self, setup: HarnessSetup) -> list[str]:
        return [
            self.executable,
            "--dangerously-skip-permissions",
            "--mcp-config",
            self._mcp_path(setup.home, setup.posix),
            "--strict-mcp-config",
        ]

    def headless_argv(self, setup: HarnessSetup, prompt: str, *, goal: bool, resume: str | None) -> list[str]:
        if resume:
            text = prompt  # continuation message; the goal is restored on --resume
        elif goal:
            text = f"/goal {prompt}"  # native goal loop runs in-process until the condition is met
        else:
            text = prompt
        argv = [
            self.executable,
            "-p",
            text,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--mcp-config",
            self._mcp_path(setup.home, setup.posix),
            "--strict-mcp-config",
        ]
        if resume:
            argv += ["--resume", resume]
        return argv

    # -- parsing ------------------------------------------------------------------------------
    def parse(self, envelope: dict[str, Any]) -> list[Event]:
        ts = float(envelope.get("ts", 0.0))
        data = envelope.get("data")
        if not isinstance(data, dict):
            text = envelope.get("text")
            return [Event("log", {"text": str(text)}, ts=ts)] if text else []

        etype = data.get("type")
        if etype == "system":
            return self._parse_system(data, ts)
        if etype == "assistant":
            return self._parse_assistant(data, ts)
        if etype == "user":
            return self._parse_user(data, ts)
        if etype == "result":
            return self._parse_result(data, ts)
        if etype == "stream_event":
            return []  # only with --include-partial-messages, which we never pass
        return []

    def _parse_system(self, data: dict[str, Any], ts: float) -> list[Event]:
        subtype = data.get("subtype")
        if subtype == "init":
            return [
                Event(
                    "session",
                    {
                        "session_id": data.get("session_id"),
                        "model": data.get("model"),
                        "cwd": data.get("cwd"),
                        "tools": data.get("tools"),
                    },
                    ts=ts,
                )
            ]
        if subtype == "compact_boundary":
            meta = data.get("compact_metadata") or {}
            return [
                Event(
                    "compaction",
                    {"pre_tokens": meta.get("pre_tokens"), "post_tokens": meta.get("post_tokens")},
                    ts=ts,
                )
            ]
        if subtype in ("task_started", "task_updated", "task_notification"):
            agent_id = data.get("agent_id") or data.get("task_id") or data.get("id")
            status = {
                "task_started": "started",
                "task_updated": data.get("status", "started"),
                "task_notification": data.get("status", "completed"),
            }[subtype]
            payload = {"id": agent_id, "status": status}
            if data.get("description"):
                payload["description"] = data["description"]
            if data.get("result") is not None:
                payload["result"] = data["result"]
            return [Event("subagent", payload, agent=_as_str(agent_id), ts=ts)]
        return []

    def _parse_assistant(self, data: dict[str, Any], ts: float) -> list[Event]:
        message = data.get("message") or {}
        agent = _as_str(data.get("agent_id"))
        events: list[Event] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                events.append(Event("text", {"text": block.get("text", "")}, agent=agent, ts=ts))
            elif btype == "thinking":
                events.append(Event("thinking", {"text": block.get("thinking", "")}, agent=agent, ts=ts))
            elif btype == "tool_use":
                events.append(
                    Event(
                        "tool_call",
                        {"id": block.get("id"), "name": block.get("name"), "input": block.get("input") or {}},
                        agent=agent,
                        ts=ts,
                    )
                )
        usage = message.get("usage")
        if isinstance(usage, dict) and usage.get("output_tokens") is not None:
            events.append(Event("usage", _usage(usage), agent=agent, ts=ts))
        return events

    def _parse_user(self, data: dict[str, Any], ts: float) -> list[Event]:
        message = data.get("message") or {}
        agent = _as_str(data.get("agent_id"))
        content = message.get("content")
        events: list[Event] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    events.append(
                        Event(
                            "tool_result",
                            {
                                "id": block.get("tool_use_id"),
                                "output": block.get("content"),
                                "is_error": bool(block.get("is_error", False)),
                            },
                            agent=agent,
                            ts=ts,
                        )
                    )
        return events

    def _parse_result(self, data: dict[str, Any], ts: float) -> list[Event]:
        events: list[Event] = []
        if data.get("session_id"):
            events.append(Event("session", {"session_id": data.get("session_id")}, ts=ts))
        usage = data.get("usage")
        if isinstance(usage, dict):
            events.append(Event("usage", _usage(usage), ts=ts))
        is_error = bool(data.get("is_error", False))
        events.append(
            Event(
                "turn_end",
                {
                    "result": data.get("result"),
                    "is_error": is_error,
                    "num_turns": data.get("num_turns"),
                    "subtype": data.get("subtype"),
                },
                ts=ts,
            )
        )
        if is_error:
            events.append(Event("error", {"message": data.get("result") or data.get("subtype")}, ts=ts))
        return events

    # -- goal ---------------------------------------------------------------------------------
    def goal_state(self, setup: HarnessSetup, events: list[Event], session_id: str | None) -> GoalState:
        """Claude's ``/goal`` runs the whole loop inside the ``-p`` process, so the process only
        exits ``-p`` when the goal is reached. A finished run whose ``result`` subtype is
        ``success`` (and not an error) means the goal is complete; anything else means keep
        resuming (an impossible goal runs until the runner's deadline).
        """

        last_turn: Event | None = None
        for event in events:
            if event.kind == "turn_end" and event.agent is None:
                last_turn = event
        if last_turn is None:
            return "active"
        if last_turn.data.get("subtype") == "success" and not last_turn.data.get("is_error"):
            return "complete"
        return "active"


# -- module helpers ---------------------------------------------------------------------------
def _joiner(posix: bool):
    module = posixpath if posix else ntpath
    return module.join


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2) + "\n"


def _as_str(value: Any) -> str | None:
    return str(value) if value else None


def _usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Normalize Anthropic usage; context tokens include cache reads/creation (research §2.5)."""

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": input_tokens + cache_read + cache_creation,
    }
