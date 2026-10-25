"""Normalized trace events shared by every harness and sandbox.

A run's on-disk trace (``<run_dir>/trace.jsonl``) is a sequence of envelopes, one JSON object per
line::

    {"ts": 1761400000.123, "src": "harness", "data": <one parsed line of the harness's JSON stream>}
    {"ts": 1761400000.456, "src": "harness", "text": "<a non-JSON stdout/stderr line>"}
    {"ts": 1761400001.000, "src": "runner", "data": {"event": "attempt", "attempt": 2, ...}}

``Harness.parse(envelope)`` turns one envelope into zero or more ``Event``s. Readers never need to
know the harness's native format.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal[
    "session",      # data: session_id, model, cwd, tools?
    "text",         # data: text                 (assistant prose)
    "thinking",     # data: text                 (model reasoning)
    "tool_call",    # data: id, name, input
    "tool_result",  # data: id, output, is_error
    "usage",        # data: input_tokens, output_tokens, context_tokens?
    "compaction",   # data: pre_tokens?, post_tokens?
    "subagent",     # data: id, status (started|completed|failed), description?
    "goal",         # data: status (active|complete|blocked|budget_limited), reason?
    "turn_end",     # data: result?, is_error, num_turns?
    "error",        # data: message
    "status",       # data: runner lifecycle: event (start|attempt|resume|timeout|exit|done), ...
    "log",          # data: text               (non-JSON output)
]


@dataclass(frozen=True)
class Event:
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)
    #: ``None`` for the main agent, otherwise a subagent/thread id.
    agent: str | None = None
    ts: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": self.data, "agent": self.agent, "ts": self.ts}


def envelope(src: str, line: str | dict[str, Any], ts: float | None = None) -> dict[str, Any]:
    """Wrap one output line (or a runner dict) in a trace envelope."""

    stamp = time.time() if ts is None else ts
    if isinstance(line, dict):
        return {"ts": stamp, "src": src, "data": line}
    stripped = line.rstrip("\r\n")
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return {"ts": stamp, "src": src, "text": stripped}
    if isinstance(parsed, dict):
        return {"ts": stamp, "src": src, "data": parsed}
    return {"ts": stamp, "src": src, "text": stripped}


def runner_events(env: dict[str, Any]) -> list[Event]:
    """Events for runner-originated envelopes (identical for every harness)."""

    data = env.get("data") or {}
    return [Event("status", dict(data), ts=float(env.get("ts", 0.0)))]
