"""Harness registry: ``get("claude")`` / ``get("codex")``."""

from __future__ import annotations

from importlib import import_module

from ..config import AomError
from .base import GoalState, Harness, HarnessSetup, McpServer, ModelAccess, SandboxProfile

#: name -> "module:Class". Adding a harness is one module plus one line here.
HARNESSES = {
    "claude": "agents_on_modal.harness.claude:ClaudeCode",
    "codex": "agents_on_modal.harness.codex:Codex",
}


def get(name: str | Harness) -> Harness:
    if isinstance(name, Harness):
        return name
    try:
        module_name, class_name = HARNESSES[name].split(":")
    except KeyError:
        raise AomError(f"Unknown harness {name!r}. Known: {', '.join(HARNESSES)}") from None
    return getattr(import_module(module_name), class_name)()


__all__ = ["HARNESSES", "GoalState", "Harness", "HarnessSetup", "McpServer", "ModelAccess", "SandboxProfile", "get"]
