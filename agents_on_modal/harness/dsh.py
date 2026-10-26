"""DeepSeek Harness (``dsh``) harness, driven through its Python SDK.

Unlike Claude Code and Codex -- CLIs that already stream newline-delimited JSON on stdout -- the
DeepSeek Harness ships as a Python SDK (``deepseek_harness.DeepSeekHarness``) that drives the ``dsh``
process over JSON-RPC and surfaces results through Python objects and an ``on_notification``
callback. The in-sandbox runner, however, launches a harness as a subprocess and parses one JSON
object per stdout line (see :mod:`agents_on_modal.runner` / :mod:`agents_on_modal.traces`). The small
adapter :mod:`agents_on_modal.harness._dsh_adapter` bridges the two: it constructs the SDK, re-emits
every notification as a line in a stable normalized schema this module owns, and prints a final
``turn_end`` from the ``RunResult``. All SDK contact lives in the adapter, so this class stays pure
and offline-testable (it never imports ``deepseek_harness``).

Wire facts (deepseek-harness-sdk 0.2.1rc2, docs ``deepseek-harness.github.io``):

* The SDK talks OpenAI chat-completions to its provider, so the gateway is reached at
  ``<base_url>/chat/completions`` -- ``protocol = "chat"``. ``base_url`` / ``api_key`` constructor
  args override ``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_API_KEY`` in the child environment; we set both.
* ``dsh_home`` is mandatory and never discovered from ``~/.dsh`` -- it is the run's private harness
  home. ``cwd`` is the agent workspace. The ``sdk-minimal`` profile is self-contained: a persistent
  shell, local execution, resumable JSONL sessions and automatic compaction. The shell is all the
  task needs -- it runs ``agent-browser`` (the browser), ``pip`` (dev packages) and writes files --
  so no extra tool mounts are required.
* dsh has no documented long-horizon "goal" primitive, so goal mode is driven exactly like Codex:
  the runner resumes the session with :meth:`continuation_prompt` until the model emits a standalone
  sentinel line, detected in :meth:`goal_state` (tool-independent, so it works on any open model).

The sandbox's root instructions are not a native dsh config file, so the adapter prepends them (and
the portable skills) to the *first* prompt only; resumes carry the restored session and so omit them.
"""

from __future__ import annotations

import os
import posixpath
from typing import Any

from ..traces import Event
from .base import GoalState, Harness, HarnessSetup, ModelAccess, SandboxProfile

#: Normalized line schema emitted by ``_dsh_adapter`` and consumed by :meth:`DeepSeek.parse`. The
#: adapter owns the SDK-facing translation; ``parse`` only maps ``data["type"]`` onto an ``Event``.
#: Keep these two in lockstep. Types: session | text | thinking | tool_call | tool_result | usage |
#: turn_end | error | subagent | log | raw (raw is debug-only and ignored by parse).

#: Default dsh profile. ``sdk-minimal`` is standalone (shell + local exec + JSONL sessions +
#: auto-compaction); the shell covers browser/pip/file-writes. Overridable via ``AOM_DSH_PROFILE``.
DEFAULT_PROFILE = "sdk-minimal"
#: Provider route. ``deepseek-official`` speaks chat-completions; with ``base_url`` pointed at the
#: gateway it posts to our ``/v1/chat/completions``. Overridable via ``AOM_DSH_PROVIDER``.
DEFAULT_PROVIDER = "deepseek-official"

#: Goal sentinels (identical contract to the Codex harness): a standalone line, so merely mentioning
#: the token does not end the goal. See :meth:`goal_state`.
GOAL_DONE_TOKEN = "GOAL_COMPLETE"
GOAL_BLOCKED_TOKEN = "GOAL_BLOCKED"

_GOAL_PREAMBLE = (
    "Treat this as a long-horizon goal: keep working across turns until every requirement is met "
    "and verified. When -- and only when -- the goal is fully achieved and you have verified it "
    "with a requirement-by-requirement audit, end your final message with a line containing "
    f"exactly {GOAL_DONE_TOKEN} and nothing else on that line. If you become permanently blocked "
    f"and cannot make progress, end with a line containing exactly {GOAL_BLOCKED_TOKEN}."
)

#: A shared, conservatively-autoscaled backend may hold a request through a cold start or a brief
#: queue; give the SDK an hours-long per-request ceiling rather than abort a slow-but-valid response.
_REQUEST_TIMEOUT_SECONDS = "7200"


class DeepSeek(Harness):
    """DeepSeek Harness (``dsh``) via its Python SDK, speaking chat-completions to the gateway."""

    name = "dsh"
    protocol = "chat"
    executable = "dsh"
    #: Pinned pip spec baked into sandbox images (the SDK vendors the ``dsh`` runtime + binary).
    pip_package = "deepseek-harness-sdk==0.2.1rc2"
    #: How the runner invokes the adapter (``python3`` resolves the baked, SDK-equipped interpreter).
    adapter_module = "agents_on_modal.harness._dsh_adapter"

    # -- preparation --------------------------------------------------------------------------
    def prepare(self, access: ModelAccess, profile: SandboxProfile, *, home: str, workdir: str) -> HarnessSetup:
        join = _joiner(profile.posix)
        instructions_path = join(home, "instructions.md")

        base_url = access.base_url.rstrip("/") + "/v1"  # the provider appends /chat/completions
        env = {
            # dsh home is mandatory and never auto-discovered; this run's private harness dir.
            "DSH_HOME": home,
            # The SDK reads these as provider credentials; base_url/api_key ctor args also set them.
            "DEEPSEEK_BASE_URL": base_url,
            "DEEPSEEK_API_KEY": access.token,
            # The adapter's own namespace (explicit ctor args, so nothing depends on env pickup).
            "AOM_DSH_BASE_URL": base_url,
            "AOM_DSH_API_KEY": access.token,
            "AOM_DSH_MODEL": access.model_id,
            "AOM_DSH_PROVIDER": DEFAULT_PROVIDER,
            "AOM_DSH_PROFILE": DEFAULT_PROFILE,
            "AOM_DSH_EFFORT": access.default_effort,
            "AOM_DSH_MAX_TOKENS": str(access.max_output_tokens),
            "AOM_DSH_WORKDIR": workdir,
            "AOM_DSH_REQUEST_TIMEOUT": _REQUEST_TIMEOUT_SECONDS,
            "AOM_DSH_INSTRUCTIONS_FILE": instructions_path,
        }
        # Opt-in debug: when this (or AOM_DSH_SEND_EFFORT) is set where the spec is built, propagate
        # it so the in-sandbox adapter dumps raw notifications / forwards the effort. Off by default,
        # so offline tests and production runs are unaffected.
        for passthrough in ("AOM_DSH_DEBUG", "AOM_DSH_SEND_EFFORT"):
            if os.environ.get(passthrough):
                env[passthrough] = os.environ[passthrough]
        env.update(profile.env)  # sandbox-supplied extras win (e.g. PATH tweaks, relay env)

        files: dict[str, str] = {
            # Root instructions + portable skills folded into one markdown the adapter prepends to
            # the first prompt (dsh has no native user-instructions file). Written under home.
            instructions_path: self._instructions_doc(profile),
        }
        # Skills also written on disk for parity / inspection (dsh sdk-minimal does not auto-load
        # them, which is why their text is also folded into the prepended instructions above).
        for skill_name, skill_text in profile.skills.items():
            files[join(home, "skills", skill_name, "SKILL.md")] = skill_text

        return HarnessSetup(
            harness=self.name, home=home, workdir=workdir, env=env, files=files, posix=profile.posix
        )

    def _instructions_doc(self, profile: SandboxProfile) -> str:
        """The sandbox instructions, plus each skill appended under its own heading.

        dsh ``sdk-minimal`` exposes no system-prompt/instructions config, so everything the agent
        must know is prepended to the first user prompt by the adapter. Folding the skills in here
        guarantees the browser recipe (and any others) reach the model even though dsh will not read
        the on-disk ``skills/`` tree.
        """

        parts = [profile.instructions.rstrip()]
        for skill_name, skill_text in profile.skills.items():
            parts.append(f"## Skill: {skill_name}\n\n{skill_text.strip()}")
        return "\n\n".join(parts) + "\n"

    # -- argv ---------------------------------------------------------------------------------
    def _adapter_base(self) -> list[str]:
        # python3 (not self.executable): the adapter drives the SDK, which spawns the dsh binary.
        return ["python3", "-m", self.adapter_module]

    def interactive_argv(self, setup: HarnessSetup) -> list[str]:
        # Best-effort TUI: dsh reads the profile + DEEPSEEK_*/DSH_HOME from the env prepared above.
        return [self.executable, "--profile", setup.env.get("AOM_DSH_PROFILE", DEFAULT_PROFILE)]

    def headless_argv(self, setup: HarnessSetup, prompt: str, *, goal: bool, resume: str | None) -> list[str]:
        """Run the adapter once. The adapter owns prompt composition (instructions prepend on the
        first turn, goal preamble), so the message is passed through verbatim as the positional.
        """

        argv = [*self._adapter_base(), "--message", prompt]
        if resume:
            argv += ["--resume", resume]
        if goal:
            argv.append("--goal")
        return argv

    def continuation_prompt(self, *, goal: bool) -> str:
        if goal:
            return (
                "Continue working toward the goal. Keep going across turns until every requirement "
                "is met and verified. When it is fully met and verified, end your message with a "
                f"line containing exactly {GOAL_DONE_TOKEN}. If you are permanently blocked, end "
                f"with a line containing exactly {GOAL_BLOCKED_TOKEN}. Otherwise keep working."
            )
        return "Continue."

    def goal_preamble(self) -> str:
        """The text the adapter appends to the first prompt in goal mode (kept here so the sentinel
        contract lives in one place)."""

        return _GOAL_PREAMBLE

    # -- parsing ------------------------------------------------------------------------------
    def parse(self, envelope: dict[str, Any]) -> list[Event]:
        """Map one adapter line (the normalized schema this module owns) onto events.

        The adapter emits well-formed JSON objects tagged by ``type``; a non-JSON line (a stray
        print from the SDK or the dsh subprocess) arrives as ``text`` on the envelope and becomes a
        ``log`` so nothing is ever silently dropped.
        """

        ts = float(envelope.get("ts", 0.0))
        data = envelope.get("data")
        if not isinstance(data, dict):
            text = envelope.get("text")
            return [Event("log", {"text": str(text)}, ts=ts)] if text else []

        etype = data.get("type")
        agent = _as_str(data.get("agent"))
        if etype == "session":
            return [Event("session", {"session_id": data.get("session_id")}, ts=ts)]
        if etype == "text":
            return [Event("text", {"text": data.get("text", "")}, agent=agent, ts=ts)]
        if etype == "thinking":
            return [Event("thinking", {"text": data.get("text", "")}, agent=agent, ts=ts)]
        if etype == "tool_call":
            return [Event("tool_call", {"id": data.get("id"), "name": data.get("name"),
                                        "input": data.get("input") or {}}, agent=agent, ts=ts)]
        if etype == "tool_result":
            return [Event("tool_result", {"id": data.get("id"), "output": data.get("output"),
                                          "is_error": bool(data.get("is_error", False))}, agent=agent, ts=ts)]
        if etype == "usage":
            return [Event("usage", {"input_tokens": int(data.get("input_tokens", 0) or 0),
                                    "output_tokens": int(data.get("output_tokens", 0) or 0)}, agent=agent, ts=ts)]
        if etype == "subagent":
            payload = {"id": data.get("id"), "status": data.get("status", "started")}
            if data.get("description"):
                payload["description"] = data["description"]
            if data.get("result") is not None:
                payload["result"] = data["result"]
            return [Event("subagent", payload, agent=agent, ts=ts)]
        if etype == "turn_end":
            events = [Event("turn_end", {"result": data.get("result"),
                                         "is_error": bool(data.get("is_error", False)),
                                         "finish_reason": data.get("finish_reason")}, ts=ts)]
            if data.get("is_error"):
                events.append(Event("error", {"message": data.get("result") or data.get("finish_reason")}, ts=ts))
            return events
        if etype == "error":
            return [Event("error", {"message": data.get("message")}, agent=agent, ts=ts)]
        if etype == "log":
            return [Event("log", {"text": data.get("text", "")}, ts=ts)]
        # type == "raw" (adapter debug dump) and anything unknown: not a normalized event.
        return []

    # -- goal ---------------------------------------------------------------------------------
    def goal_state(self, setup: HarnessSetup, events: list[Event], session_id: str | None) -> GoalState:
        """Terminal when the model's own message ends with a standalone sentinel line. Identical in
        spirit to the Codex harness: the last terminal signal in order wins, and a bare mention of
        the token does not count (it must stand alone on its line)."""

        last_status: str | None = None
        for event in events:
            if event.kind != "text":
                continue
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
    if posix:
        return posixpath.join
    import ntpath

    return ntpath.join


def _as_str(value: Any) -> str | None:
    return str(value) if value else None
