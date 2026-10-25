"""Map a client reasoning effort + the model's ``Reasoning`` config onto chat-completions fields.

The per-family mapping (e.g. Qwen3.8 ``high`` -> engine ``xhigh``, DeepSeek ``medium`` -> ``high``)
lives in the catalog's ``Reasoning.efforts`` table; this module only applies it. The input is the
JSON dict shape of ``agents_on_modal.models.Reasoning`` (``{"default", "efforts", "kwarg"}``).
"""

from __future__ import annotations

from typing import Any

# ``agents_on_modal.models`` is modal-free (Modal is imported lazily), so this is safe at load.
from agents_on_modal.models import normalize_effort


def _on(kwarg: str | None, effort: str | None) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if effort is not None:
        fields["reasoning_effort"] = effort
    if kwarg:
        fields["chat_template_kwargs"] = {kwarg: True}
    return fields


def _off(kwarg: str | None) -> dict[str, Any]:
    if kwarg:
        return {"chat_template_kwargs": {kwarg: False}}
    # No template toggle: ``reasoning_effort="none"`` turns thinking off on SGLang/llama.cpp.
    return {"reasoning_effort": "none"}


def chat_reasoning_fields(
    effort: str | None, reasoning: dict[str, Any] | None, *, enabled: bool | None = None
) -> dict[str, Any]:
    """Chat-body fields for the given client effort.

    ``reasoning`` is the ``Reasoning`` dict. ``enabled`` is the client's explicit thinking toggle:
    ``False`` forces thinking off, ``True``/``None`` leave the per-effort mapping in charge.

    Returns some subset of ``{"reasoning_effort": str, "chat_template_kwargs": {kwarg: bool}}``.
    """

    reasoning = reasoning or {}
    efforts: dict[str, Any] = reasoning.get("efforts") or {}
    default: str = reasoning.get("default") or "high"
    kwarg: str | None = reasoning.get("kwarg")

    if enabled is False:
        return _off(kwarg)

    normalized = normalize_effort(effort) or default
    if normalized in efforts:
        value = efforts[normalized]
    elif default in efforts:
        value = efforts[default]
    else:
        value = True

    if value is False:
        # The effort maps to "off" for this family; honour an explicit on by reasoning at default.
        return _on(kwarg, None) if enabled else _off(kwarg)
    if value is True:
        return _on(kwarg, None)
    return _on(kwarg, str(value))
