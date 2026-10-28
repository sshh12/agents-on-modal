"""The gateway's internal representation (IR).

Both client protocols (Anthropic Messages, OpenAI Responses) are normalised into the same request
IR -- ``Message``s made of typed ``Part``s, plus ``Tool``s and sampling knobs -- which is then
rendered into a single chat-completions body. Upstream chat SSE is parsed into a flat stream of
IR events (``ReasoningDelta`` / ``TextDelta`` / ``ToolCallStart`` / ``ToolArgsDelta`` / ``Finish`` /
``UsageEvent`` / ``ErrorEvent``) which each protocol's SSE writer consumes.

This module is pure stdlib; it never imports aiohttp or modal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Union

# --------------------------------------------------------------------------------------------------
# Request IR
# --------------------------------------------------------------------------------------------------


@dataclass
class ToolCall:
    """An assistant tool call. ``kind`` is ``function`` or ``custom``; ``namespace`` is set for
    Codex namespace tools so the flattened ``ns__member`` name can be restored on output."""

    id: str
    name: str
    args_json: str
    kind: str = "function"
    namespace: str | None = None


@dataclass
class Tool:
    """A tool offered to the model. ``kind`` records the client-side flavour (``function`` |
    ``custom`` | ``namespace`` member) so results can be mapped back."""

    name: str
    description: str
    parameters: dict[str, Any]
    kind: str = "function"
    namespace: str | None = None
    member: str | None = None

    def to_chat(self) -> dict[str, Any]:
        fn: dict[str, Any] = {"name": self.name, "parameters": self.parameters}
        if self.description:
            fn["description"] = self.description
        return {"type": "function", "function": fn}


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    url: str


@dataclass
class ReasoningPart:
    text: str


@dataclass
class ToolCallPart:
    call: ToolCall


@dataclass
class ToolResultPart:
    id: str
    content: Any
    is_error: bool = False


Part = Union[TextPart, ImagePart, ReasoningPart, ToolCallPart, ToolResultPart]


@dataclass
class Message:
    """One conversation message. ``role`` is ``system`` | ``user`` | ``assistant``.

    (A mid-conversation ``system`` message is relocated by :func:`to_chat_messages`.)
    """

    role: str
    parts: list[Part] = field(default_factory=list)


@dataclass
class ParsedRequest:
    """A client request normalised into chat-ready pieces plus stream-translation context.

    The server assembles the final chat body (model id, reasoning mapping, ``max_tokens`` clamp,
    stream options) from these fields; the SSE writers use the context fields.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    stop: list[str] = field(default_factory=list)
    sampling: dict[str, Any] = field(default_factory=dict)
    parallel_tool_calls: bool | None = None
    effort: str | None = None
    thinking_enabled: bool | None = None
    max_tokens_requested: int | None = None
    #: Flattened names the model sees as functions but that must be emitted as Codex custom tools.
    custom_tool_names: set[str] = field(default_factory=set)
    #: Flattened ``ns__member`` name -> namespace, for restoring Codex namespace tool calls.
    namespaces: dict[str, str] = field(default_factory=dict)
    #: Responses only: whether to emit reasoning summary events.
    summary_enabled: bool = False


# --------------------------------------------------------------------------------------------------
# Usage + stream-event IR
# --------------------------------------------------------------------------------------------------


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def input_tokens_ex_cache(self) -> int:
        """Anthropic ``input_tokens`` excludes cache reads; OpenAI ``input_tokens`` includes them."""

        return max(self.prompt_tokens - self.cached_tokens, 0)


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallStart:
    index: int
    id: str
    name: str


@dataclass
class ToolArgsDelta:
    index: int
    delta: str


@dataclass
class Finish:
    reason: str | None
    stop_matched: str | None = None


@dataclass
class UsageEvent:
    usage: Usage


@dataclass
class ErrorEvent:
    code: str
    message: str
    status: int = 500


StreamEvent = Union[
    ReasoningDelta, TextDelta, ToolCallStart, ToolArgsDelta, Finish, UsageEvent, ErrorEvent
]


# --------------------------------------------------------------------------------------------------
# Rendering the request IR into a chat-completions ``messages`` array
# --------------------------------------------------------------------------------------------------

_COMPACT = (",", ":")


def dumps_args(value: Any) -> str:
    """Deterministic JSON for tool-call arguments (stable turn to turn for the prefix cache)."""

    return json.dumps(value, ensure_ascii=False, separators=_COMPACT)


def _collapse_content(content: Any, *, is_error: bool = False) -> Any:
    """A tool-result / block content value collapsed to a chat string when text-only."""

    if isinstance(content, str):
        text = content
        return f"Error: {text}" if is_error else text
    if isinstance(content, list):
        has_image = any(isinstance(b, dict) and b.get("type") in ("image", "image_url") for b in content)
        if not has_image:
            texts = []
            for block in content:
                if isinstance(block, dict):
                    texts.append(str(block.get("text", "")))
                else:
                    texts.append(str(block))
            text = "\n".join(t for t in texts if t)
            return f"Error: {text}" if is_error else text
        return content  # keep the array when images are present
    if content is None:
        return "Error" if is_error else ""
    return str(content)


def _text_of(message: Message) -> str:
    return "".join(p.text for p in message.parts if isinstance(p, TextPart))


def _reasoning_of(message: Message) -> str:
    return "".join(p.text for p in message.parts if isinstance(p, ReasoningPart))


def _user_content(message: Message) -> Any:
    """``str`` for text-only user turns, else a content-block list; ``None`` when empty."""

    texts = [p.text for p in message.parts if isinstance(p, TextPart)]
    images = [p for p in message.parts if isinstance(p, ImagePart)]
    if not images:
        joined = "\n".join(t for t in texts if t)
        return joined or None
    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart) and part.text:
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image_url", "image_url": {"url": part.url}})
    return blocks or None


def _tool_message(result: ToolResultPart) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": result.id,
        "content": _collapse_content(result.content, is_error=result.is_error),
    }


def _assistant_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant"}
    text = _text_of(message)
    reasoning = _reasoning_of(message)
    tool_calls = []
    for part in message.parts:
        if isinstance(part, ToolCallPart):
            call = part.call
            tool_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.args_json},
                }
            )
    out["content"] = text if text else None
    if reasoning:
        out["reasoning_content"] = reasoning
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _append_reminder(out: list[dict[str, Any]], text: str) -> None:
    reminder = f"<system-reminder>\n{text}\n</system-reminder>"
    if out and out[-1]["role"] == "user":
        content = out[-1]["content"]
        if isinstance(content, str):
            out[-1]["content"] = content + "\n\n" + reminder
            return
        if isinstance(content, list):
            content.append({"type": "text", "text": reminder})
            return
    out.append({"role": "user", "content": reminder})


def to_chat_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Render the request IR into a chat-completions ``messages`` array.

    * Only the leading ``system`` string becomes the chat system message.
    * Mid-conversation ``system`` messages are appended to the preceding user turn as a
      ``<system-reminder>`` block (a fresh user turn when the predecessor is not a user message).
    * Tool results become ``tool`` messages emitted before the same user turn's text.
    """

    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "system":
            text = _text_of(message)
            if text:
                _append_reminder(out, text)
            continue
        if message.role == "user":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    out.append(_tool_message(part))
            content = _user_content(message)
            if content is not None:
                out.append({"role": "user", "content": content})
            continue
        if message.role == "assistant":
            out.append(_assistant_message(message))
    return out


#: Placeholder that replaces an image block when the served model is text-only (``vision: false``).
IMAGE_OMITTED = "[image omitted: this model is text-only]"


def drop_images_for_text_model(messages: list[dict[str, Any]]) -> int:
    """Replace image blocks with a text placeholder in already-rendered chat ``messages`` (in place).

    A text-only model (``vision: false``) cannot accept image content. The harnesses still put image
    blocks in a request -- a pasted screenshot, a browser tool result, or, most insidiously, an older
    turn that auto-compaction folds back in -- and the engine then rejects the *whole* request
    (``content should be a valid string`` / ``validate_thinking_parts_role``), killing the run deep
    into a long context (issue S1). Stripping them here, at the gateway right before the engine, fixes
    it regardless of how the image got into the history. Idempotent (a second pass is a no-op, since
    the content is a plain string by then). Returns how many messages had an image replaced.
    """

    changed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if not any(isinstance(b, dict) and b.get("type") in ("image", "image_url") for b in content):
            continue
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                parts.append(IMAGE_OMITTED)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
        msg["content"] = "\n".join(p for p in parts if p) or IMAGE_OMITTED
        changed += 1
    return changed


def estimate_tokens(text: str) -> int:
    """A cheap, deterministic prompt-size estimate (~4 chars/token) for ``max_tokens`` clamping."""

    return int(math.ceil(len(text) / 4)) if text else 0


def clamp_max_tokens(
    requested: int | None,
    *,
    context_window: int,
    prompt_estimate: int,
    model_max_output: int,
    default: int,
    margin: int = 64,
) -> int:
    """``max_tokens`` for the engine: never exceed the model cap or what the context window allows.

    SGLang rejects ``input + max_new_tokens > context_length``, so the headroom
    ``context_window - prompt_estimate - margin`` is a hard ceiling. The result is at least 1.
    """

    want = requested if requested and requested > 0 else default
    headroom = context_window - prompt_estimate - margin
    ceiling = min(model_max_output, headroom)
    return max(1, min(want, ceiling))
