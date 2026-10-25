"""The protocol-translation gateway.

Runs inside the model's GPU container next to the engine and speaks every harness protocol
(Anthropic Messages, OpenAI Responses, chat completions) on top of the engine's streaming
``/v1/chat/completions``. The public entry points are :func:`build_app` and :func:`run`.

Imports here are stdlib + aiohttp + the modal-free ``agents_on_modal.models`` only.
"""

from __future__ import annotations

from .server import build_app, run

__all__ = ["build_app", "run"]
