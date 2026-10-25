"""Sandbox registry.

    from agents_on_modal import sandbox
    box = sandbox.create("docker", proxy="socks5h://user:pass@host:1080")
    run = box.start("claude", endpoint, "fix the failing tests", goal=True, timeout=3600)
    for event in run.events(follow=True):
        print(event.kind, event.data)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ..config import AomError
from .base import ExecResult, Run, Sandbox, SandboxError, new_id

#: kind -> "module:Class".
SANDBOXES = {
    "local": "agents_on_modal.sandbox.local:LocalSandbox",
    "docker": "agents_on_modal.sandbox.docker:DockerSandbox",
    "modal": "agents_on_modal.sandbox.modal:ModalSandbox",
}


def kind(name: str) -> type[Sandbox]:
    try:
        module_name, class_name = SANDBOXES[name].split(":")
    except KeyError:
        raise AomError(f"Unknown sandbox kind {name!r}. Known: {', '.join(SANDBOXES)}") from None
    return getattr(import_module(module_name), class_name)


def create(name: str, **options: Any) -> Sandbox:
    return kind(name).create(**options)


def get(name: str, sandbox_id: str) -> Sandbox:
    return kind(name).get(sandbox_id)


def list_sandboxes(name: str, **filters: Any) -> list[Sandbox]:
    return kind(name).list(**filters)


__all__ = ["SANDBOXES", "ExecResult", "Run", "Sandbox", "SandboxError", "create", "get", "kind", "list_sandboxes", "new_id"]
