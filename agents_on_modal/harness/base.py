"""The harness contract: turn (model access, sandbox profile, home dir) into files + env + argv.

A harness never touches a sandbox directly. ``Sandbox.setup_harness`` calls ``prepare`` and writes
the returned files; the in-sandbox runner calls ``headless_argv`` / ``parse`` / ``goal_state``. That
split keeps every harness testable as pure functions plus a few local-file reads.
"""

from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from ..traces import Event

GoalState = Literal["complete", "active", "blocked", "none"]


@dataclass(frozen=True)
class ModelAccess:
    """The model as reachable from wherever the harness process runs.

    ``base_url`` is the gateway root (no ``/v1``); the gateway serves ``/v1/messages``,
    ``/v1/responses`` and ``/v1/chat/completions`` under it. ``token`` is what the harness sends as a
    bearer token; it may be a placeholder when a broker injects the real credential.
    """

    base_url: str
    token: str
    model_id: str
    context_window: int
    max_output_tokens: int
    default_effort: str = "high"
    vision: bool = False

    @classmethod
    def from_endpoint(cls, endpoint: Any, *, base_url: str | None = None, token: str | None = None) -> "ModelAccess":
        model = endpoint.model
        return cls(
            base_url=(base_url or endpoint.url).rstrip("/"),
            token=token if token is not None else endpoint.token,
            model_id=model.name,
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            default_effort=model.reasoning.default,
            vision=model.vision,
        )


@dataclass(frozen=True)
class McpServer:
    """A stdio MCP server the sandbox provides (e.g. Playwright for the browser)."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxProfile:
    """What a sandbox tells every harness about itself."""

    kind: str                                   # local | docker | modal
    instructions: str                           # root instructions text (markdown)
    posix: bool = True                          # harness runs on a POSIX filesystem
    skills: dict[str, str] = field(default_factory=dict)        # skill name -> SKILL.md text
    mcp_servers: dict[str, McpServer] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)           # extra env for the harness process


@dataclass
class HarnessSetup:
    """Everything needed to launch a prepared harness. JSON-serializable apart from ``files``."""

    harness: str
    home: str                                   # harness home dir (CLAUDE_CONFIG_DIR / CODEX_HOME)
    workdir: str                                # cwd for the harness process
    env: dict[str, str]                         # complete env additions for the harness process
    files: dict[str, str] = field(default_factory=dict)         # absolute path -> text content
    posix: bool = True

    def to_json(self) -> dict[str, Any]:
        return {"harness": self.harness, "home": self.home, "workdir": self.workdir,
                "env": self.env, "posix": self.posix}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "HarnessSetup":
        return cls(harness=data["harness"], home=data["home"], workdir=data["workdir"],
                   env=dict(data["env"]), posix=bool(data.get("posix", True)))

    def path(self, *parts: str) -> str:
        """Join below ``home`` using the target filesystem's separator."""

        if self.posix:
            return posixpath.join(self.home, *parts)
        import ntpath

        return ntpath.join(self.home, *parts)


class Harness(ABC):
    """One agent CLI. Subclasses are stateless."""

    #: Registry key, e.g. ``claude``.
    name: str = ""
    #: Wire protocol the CLI speaks to the model: ``anthropic`` | ``responses`` | ``chat``.
    protocol: str = ""
    #: The executable to run.
    executable: str = ""
    #: Pinned npm install spec baked into sandbox images.
    npm_package: str = ""

    @abstractmethod
    def prepare(self, access: ModelAccess, profile: SandboxProfile, *, home: str, workdir: str) -> HarnessSetup:
        """Config files + env for a maximally permissive, single-model, isolated harness home."""

    @abstractmethod
    def interactive_argv(self, setup: HarnessSetup) -> list[str]:
        """argv for an interactive TUI session in ``setup.workdir``."""

    @abstractmethod
    def headless_argv(self, setup: HarnessSetup, prompt: str, *, goal: bool, resume: str | None) -> list[str]:
        """argv for one headless invocation that streams JSON events to stdout.

        ``resume`` is a session id from a previous invocation; when set, ``prompt`` is the
        continuation message. ``goal`` asks the harness to treat ``prompt`` as a long-horizon goal.
        """

    def continuation_prompt(self, *, goal: bool) -> str:
        """The message sent when the runner resumes a session (goal mode or after a crash)."""

        if goal:
            return "Continue working toward the active goal. Do not stop until it is fully achieved and verified."
        return "Continue."

    @abstractmethod
    def parse(self, envelope: dict[str, Any]) -> list[Event]:
        """Normalize one trace envelope (see ``agents_on_modal.traces``) into events."""

    def session_id(self, events: list[Event]) -> str | None:
        """The resumable session id seen in ``events`` (last ``session`` event wins)."""

        found = None
        for event in events:
            if event.kind == "session" and event.agent is None and event.data.get("session_id"):
                found = str(event.data["session_id"])
        return found

    @abstractmethod
    def goal_state(self, setup: HarnessSetup, events: list[Event], session_id: str | None) -> GoalState:
        """Whether the goal of the run that produced ``events`` is complete.

        Called by the in-sandbox runner after each invocation exits; it may read files under
        ``setup.home`` (the runner shares the harness's filesystem).
        """
