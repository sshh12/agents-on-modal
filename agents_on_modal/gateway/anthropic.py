"""Anthropic Messages <-> IR <-> chat completions.

Request parsing (``parse_request``), the streaming SSE writer (``stream_sse``), the non-streaming
response builder (``response_json``), and ``count_tokens``. Protocol details follow research
``protocols.md`` sections A and D.2/D.3.
"""

from __future__ import annotations

import json
import secrets
from typing import Any, AsyncIterator

import aiohttp

from . import chat, ir

_BILLING_PREFIX = "x-anthropic-billing-header"


def _rand(prefix: str) -> str:
    return prefix + secrets.token_hex(12)


def _frame(event_type: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# --------------------------------------------------------------------------------------------------
# Request: Anthropic -> IR -> chat body pieces
# --------------------------------------------------------------------------------------------------


def _system_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if not isinstance(block, dict):
            continue
        text = block.get("text", "")
        if isinstance(text, str) and text.startswith(_BILLING_PREFIX):
            continue  # strip the per-request billing header (prefix-cache hygiene)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _image_url(source: dict[str, Any]) -> str | None:
    kind = source.get("type")
    if kind == "base64":
        return f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    if kind == "url":
        return source.get("url")
    return None


def _message_parts(content: Any) -> list[ir.Part]:
    if isinstance(content, str):
        return [ir.TextPart(content)] if content else []
    parts: list[ir.Part] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            if block.get("text"):
                parts.append(ir.TextPart(block["text"]))
        elif btype == "image":
            url = _image_url(block.get("source") or {})
            if url:
                parts.append(ir.ImagePart(url))
        elif btype == "thinking":
            if block.get("thinking"):
                parts.append(ir.ReasoningPart(block["thinking"]))
        elif btype == "redacted_thinking":
            continue  # unrenderable on chat engines
        elif btype == "tool_use":
            parts.append(
                ir.ToolCallPart(
                    ir.ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        args_json=ir.dumps_args(block.get("input") or {}),
                    )
                )
            )
        elif btype == "tool_result":
            parts.append(
                ir.ToolResultPart(
                    id=block.get("tool_use_id", ""),
                    content=block.get("content", ""),
                    is_error=bool(block.get("is_error")),
                )
            )
    return parts


def _tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    ctype = choice.get("type")
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "none":
        return "none"
    if ctype == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def _convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("defer_loading"):
            continue  # the backend cannot lazy-load; drop the placeholder
        ttype = tool.get("type")
        if ttype not in (None, "custom"):
            continue  # server/built-in tool
        schema = dict(tool.get("input_schema") or {"type": "object"})
        schema.pop("$schema", None)
        fn: dict[str, Any] = {"name": tool.get("name", ""), "parameters": schema}
        if tool.get("description"):
            fn["description"] = tool["description"]
        if tool.get("strict") is not None:
            fn["strict"] = tool["strict"]
        out.append({"type": "function", "function": fn})
    return out


def parse_request(body: dict[str, Any]) -> ir.ParsedRequest:
    system = _system_text(body.get("system"))
    messages = [ir.Message(role=m.get("role", "user"), parts=_message_parts(m.get("content")))
                for m in body.get("messages", [])]
    chat_messages = ir.to_chat_messages(system, messages)

    output_config = body.get("output_config") or {}
    effort = output_config.get("effort")
    response_format = None
    fmt = output_config.get("format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": fmt.get("schema") or {}, "strict": True},
        }

    thinking = body.get("thinking")
    thinking_enabled: bool | None = None
    if isinstance(thinking, dict):
        thinking_enabled = thinking.get("type") != "disabled"

    sampling: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            sampling[key] = body[key]

    choice = _tool_choice(body.get("tool_choice"))
    parallel = None
    tc = body.get("tool_choice")
    if isinstance(tc, dict) and tc.get("disable_parallel_tool_use"):
        parallel = False

    return ir.ParsedRequest(
        messages=chat_messages,
        tools=_convert_tools(body.get("tools") or []),
        tool_choice=choice,
        response_format=response_format,
        stop=list(body.get("stop_sequences") or []),
        sampling=sampling,
        parallel_tool_calls=parallel,
        effort=effort,
        thinking_enabled=thinking_enabled,
        max_tokens_requested=body.get("max_tokens"),
    )


# --------------------------------------------------------------------------------------------------
# Stop-reason mapping
# --------------------------------------------------------------------------------------------------


def _stop_reason(finish: str | None, stop_matched: str | None, stops: list[str], any_tool: bool) -> tuple[str, str | None]:
    if any_tool:
        return "tool_use", None
    if finish == "length":
        return "max_tokens", None
    if finish in ("content_filter", "refusal"):
        return "refusal", None
    if finish == "stop" and stop_matched and stop_matched in stops:
        return "stop_sequence", stop_matched
    return "end_turn", None


_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def error_payload(exc: Exception, *, context_window: int) -> tuple[int, dict[str, Any]]:
    """HTTP status + Anthropic error envelope for an upstream failure."""

    if isinstance(exc, chat.ContextOverflow):
        prompt = exc.prompt_tokens if exc.prompt_tokens is not None else context_window + 1
        maximum = exc.max_tokens if exc.max_tokens is not None else context_window
        message = f"prompt is too long: {prompt} tokens > {maximum} maximum"
        return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    status = getattr(exc, "status", 500)
    etype = _ERROR_TYPES.get(status, "api_error")
    return status, {"type": "error", "error": {"type": etype, "message": str(exc)}}


# --------------------------------------------------------------------------------------------------
# Streaming SSE writer
# --------------------------------------------------------------------------------------------------


async def stream_sse(
    session: aiohttp.ClientSession,
    engine_url: str,
    body: dict[str, Any],
    *,
    model_id: str,
    stops: list[str],
    context_window: int,
    keepalive: float = 10.0,
) -> AsyncIterator[bytes]:
    msg_id = _rand("msg_")
    start = {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model_id,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield _frame("message_start", start)
    yield _frame("ping", {"type": "ping"})

    state: dict[str, Any] = {"open": None, "next": 0}

    def close_block() -> list[bytes]:
        if state["open"] is None:
            return []
        _, idx = state["open"]
        state["open"] = None
        return [_frame("content_block_stop", {"type": "content_block_stop", "index": idx})]

    def open_block(kind: str, block: dict[str, Any]) -> tuple[int, bytes]:
        idx = state["next"]
        state["next"] += 1
        state["open"] = (kind, idx)
        return idx, _frame("content_block_start", {"type": "content_block_start", "index": idx, "content_block": block})

    def ensure(kind: str) -> tuple[int, list[bytes]]:
        if state["open"] and state["open"][0] == kind:
            return state["open"][1], []
        frames = close_block()
        block = {"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking" else {"type": "text", "text": ""}
        idx, f = open_block(kind, block)
        frames.append(f)
        return idx, frames

    tool_order: list[int] = []
    tool_info: dict[int, tuple[str, str]] = {}
    tool_args: dict[int, str] = {}
    streaming_tool: int | None = None
    finish_reason: str | None = None
    stop_matched: str | None = None
    usage = ir.Usage()

    source = chat.stream_events(session, engine_url, body)
    try:
        async for item in chat.stream_with_keepalive(source, keepalive):
            if item is chat.KEEPALIVE:
                yield _frame("ping", {"type": "ping"})
                continue
            if isinstance(item, ir.ReasoningDelta):
                idx, frames = ensure("thinking")
                for f in frames:
                    yield f
                yield _frame("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                      "delta": {"type": "thinking_delta", "thinking": item.text}})
            elif isinstance(item, ir.TextDelta):
                idx, frames = ensure("text")
                for f in frames:
                    yield f
                yield _frame("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                      "delta": {"type": "text_delta", "text": item.text}})
            elif isinstance(item, ir.ToolCallStart):
                tool_order.append(item.index)
                tool_info[item.index] = (item.id, item.name)
                tool_args[item.index] = ""
                if streaming_tool is None:
                    for f in close_block():
                        yield f
                    _, f = open_block("tool", {"type": "tool_use", "id": item.id, "name": item.name, "input": {}})
                    streaming_tool = item.index
                    yield f
            elif isinstance(item, ir.ToolArgsDelta):
                tool_args[item.index] = tool_args.get(item.index, "") + item.delta
                if item.index == streaming_tool and state["open"] and state["open"][0] == "tool":
                    yield _frame("content_block_delta", {"type": "content_block_delta", "index": state["open"][1],
                                                         "delta": {"type": "input_json_delta", "partial_json": item.delta}})
            elif isinstance(item, ir.Finish):
                finish_reason = item.reason
                stop_matched = item.stop_matched
            elif isinstance(item, ir.UsageEvent):
                usage = item.usage
    except Exception as exc:  # noqa: BLE001 - surface any upstream failure as an SSE error frame
        for f in close_block():
            yield f
        _, payload = error_payload(exc, context_window=context_window)
        yield _frame("error", payload)
        return

    for f in close_block():
        yield f

    # Flush buffered (parallel) tool calls that were not streamed live.
    for index in tool_order:
        if index == streaming_tool:
            continue
        call_id, name = tool_info[index]
        _, f = open_block("tool", {"type": "tool_use", "id": call_id, "name": name, "input": {}})
        yield f
        args = tool_args.get(index, "")
        if args:
            yield _frame("content_block_delta", {"type": "content_block_delta", "index": state["open"][1],
                                                 "delta": {"type": "input_json_delta", "partial_json": args}})
        for f in close_block():
            yield f

    reason, stop_seq = _stop_reason(finish_reason, stop_matched, stops, bool(tool_order))
    yield _frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reason, "stop_sequence": stop_seq},
        "usage": {
            "input_tokens": usage.input_tokens_ex_cache,
            "cache_read_input_tokens": usage.cached_tokens,
            "output_tokens": usage.completion_tokens,
        },
    })
    yield _frame("message_stop", {"type": "message_stop"})


# --------------------------------------------------------------------------------------------------
# Non-streaming response
# --------------------------------------------------------------------------------------------------


async def response_json(
    session: aiohttp.ClientSession,
    engine_url: str,
    body: dict[str, Any],
    *,
    model_id: str,
    stops: list[str],
) -> dict[str, Any]:
    result = await chat.aggregate(session, engine_url, body)
    content: list[dict[str, Any]] = []
    if result.reasoning:
        content.append({"type": "thinking", "thinking": result.reasoning, "signature": ""})
    if result.text:
        content.append({"type": "text", "text": result.text})
    for call in result.tool_calls:
        try:
            parsed = json.loads(call.args) if call.args else {}
        except ValueError:
            parsed = {}
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": parsed})
    reason, stop_seq = _stop_reason(result.finish_reason, result.stop_matched, stops, bool(result.tool_calls))
    usage = result.usage
    return {
        "id": _rand("msg_"), "type": "message", "role": "assistant", "model": model_id,
        "content": content, "stop_reason": reason, "stop_sequence": stop_seq, "stop_details": None,
        "usage": {
            "input_tokens": usage.input_tokens_ex_cache,
            "output_tokens": usage.completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": usage.cached_tokens,
        },
    }


def count_tokens(body: dict[str, Any]) -> dict[str, Any]:
    """Estimate prompt tokens for ``/v1/messages/count_tokens`` (no engine round-trip)."""

    parsed = parse_request(body)
    blob = json.dumps(parsed.messages, ensure_ascii=False) + json.dumps(parsed.tools, ensure_ascii=False)
    return {"input_tokens": ir.estimate_tokens(blob)}
