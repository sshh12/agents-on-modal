"""OpenAI Responses <-> IR <-> chat completions.

Request parsing (``parse_request``) and the streaming SSE writer (``stream_sse``). Codex builds its
history only from ``response.output_item.done`` items plus ``response.completed`` (research
``codex.md`` / ``protocols.md`` B and D.4/D.5), so tool-call items are buffered and emitted whole.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, AsyncIterator

import aiohttp

from . import chat, ir


def _rand(prefix: str) -> str:
    return prefix + secrets.token_hex(12)


def _frame(event_type: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _strip_encrypted(obj: Any) -> Any:
    """Recursively drop the non-standard ``encrypted`` schema keyword (multi_agent_v2)."""

    if isinstance(obj, dict):
        return {k: _strip_encrypted(v) for k, v in obj.items() if k != "encrypted"}
    if isinstance(obj, list):
        return [_strip_encrypted(v) for v in obj]
    return obj


# --------------------------------------------------------------------------------------------------
# Request: Responses -> IR -> chat body pieces
# --------------------------------------------------------------------------------------------------


def _content_text(content: Any) -> list[ir.Part]:
    parts: list[ir.Part] = []
    if isinstance(content, str):
        return [ir.TextPart(content)] if content else []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in ("input_text", "output_text", "text"):
            if block.get("text"):
                parts.append(ir.TextPart(block["text"]))
        elif btype == "input_image":
            url = block.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if url:
                parts.append(ir.ImagePart(url))
    return parts


def _output_to_str(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        texts = [b.get("text", "") for b in output if isinstance(b, dict) and b.get("type") in ("input_text", "output_text", "text")]
        return "\n".join(t for t in texts if t)
    return "" if output is None else str(output)


def _custom_tool(tool: dict[str, Any]) -> tuple[dict[str, Any], str]:
    name = tool.get("name", "")
    description = tool.get("description", "")
    fmt = tool.get("format") or {}
    definition = fmt.get("definition")
    if definition:
        description = f"{description}\n\nFormat:\n```{fmt.get('syntax', 'lark')}\n{definition}\n```"
    params = {
        "type": "object",
        "properties": {"input": {"type": "string", "description": f"raw {name} input in the format above"}},
        "required": ["input"],
    }
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}, name


def _function_tool(tool: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    fn: dict[str, Any] = {
        "name": name or tool.get("name", ""),
        "parameters": _strip_encrypted(tool.get("parameters") or {"type": "object"}),
    }
    if tool.get("description"):
        fn["description"] = tool["description"]
    if tool.get("strict") is not None:
        fn["strict"] = tool["strict"]
    return {"type": "function", "function": fn}


def _parse_tools(tools: list[Any]) -> tuple[list[dict[str, Any]], set[str], dict[str, str]]:
    out: list[dict[str, Any]] = []
    custom_names: set[str] = set()
    namespaces: dict[str, str] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type", "function")
        if ttype == "function":
            out.append(_function_tool(tool))
        elif ttype == "custom":
            chat_tool, name = _custom_tool(tool)
            out.append(chat_tool)
            custom_names.add(name)
        elif ttype == "namespace":
            ns = tool.get("name", "")
            ns_desc = tool.get("description", "")
            for member in tool.get("tools") or []:
                if not isinstance(member, dict):
                    continue
                flat = f"{ns}__{member.get('name', '')}"
                desc = member.get("description", "")
                if ns_desc:
                    desc = f"{ns_desc}\n\n{desc}" if desc else ns_desc
                out.append(_function_tool({**member, "description": desc}, name=flat))
                namespaces[flat] = ns
        # web_search / tool_search / everything else: dropped
    return out, custom_names, namespaces


def _tool_choice(choice: Any) -> Any:
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype in ("function", "custom") and choice.get("name"):
            return {"type": "function", "function": {"name": choice["name"]}}
    return None


def parse_request(body: dict[str, Any]) -> ir.ParsedRequest:
    system_text = body.get("instructions") or ""
    messages: list[ir.Message] = []
    pending: ir.Message | None = None
    leading = True

    extra_tools: list[Any] = []

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            messages.append(pending)
            pending = None

    def assistant() -> ir.Message:
        nonlocal pending
        if pending is None:
            pending = ir.Message(role="assistant", parts=[])
        return pending

    for item in body.get("input") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "message")
        if itype == "message":
            role = item.get("role", "user")
            if role in ("developer", "system"):
                text = "\n".join(p.text for p in _content_text(item.get("content")) if isinstance(p, ir.TextPart))
                if leading:
                    system_text = f"{system_text}\n\n{text}" if system_text else text
                elif text:
                    flush()
                    messages.append(ir.Message(role="system", parts=[ir.TextPart(text)]))
            elif role == "user":
                leading = False
                flush()
                messages.append(ir.Message(role="user", parts=_content_text(item.get("content"))))
            elif role == "assistant":
                leading = False
                assistant().parts.extend(_content_text(item.get("content")))
        elif itype == "reasoning":
            leading = False
            text = ""
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "reasoning_text":
                    text += block.get("text", "")
            if not text:
                for block in item.get("summary") or []:
                    if isinstance(block, dict):
                        text += block.get("text", "")
            if text:
                assistant().parts.append(ir.ReasoningPart(text))
        elif itype == "function_call":
            leading = False
            ns = item.get("namespace")
            name = item.get("name", "")
            flat = f"{ns}__{name}" if ns else name
            assistant().parts.append(ir.ToolCallPart(ir.ToolCall(
                id=item.get("call_id", ""), name=flat, args_json=item.get("arguments") or "{}")))
        elif itype == "custom_tool_call":
            leading = False
            assistant().parts.append(ir.ToolCallPart(ir.ToolCall(
                id=item.get("call_id", ""), name=item.get("name", ""),
                args_json=ir.dumps_args({"input": item.get("input", "")}), kind="custom")))
        elif itype in ("function_call_output", "custom_tool_call_output"):
            leading = False
            flush()
            messages.append(ir.Message(role="user", parts=[ir.ToolResultPart(
                id=item.get("call_id", ""), content=_output_to_str(item.get("output")))]))
        elif itype == "additional_tools":
            extra_tools.extend(item.get("tools") or [])

    flush()
    chat_messages = ir.to_chat_messages(system_text, messages)

    tools, custom_names, namespaces = _parse_tools((body.get("tools") or []) + extra_tools)

    reasoning = body.get("reasoning") or {}
    effort = reasoning.get("effort")
    summary = reasoning.get("summary")
    summary_enabled = bool(summary) and summary != "none"

    response_format = None
    text_cfg = body.get("text") or {}
    fmt = text_cfg.get("format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name", "output"),
                "schema": fmt.get("schema") or {},
                "strict": bool(fmt.get("strict", True)),
            },
        }

    sampling: dict[str, Any] = {}
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            sampling[key] = body[key]

    return ir.ParsedRequest(
        messages=chat_messages,
        tools=tools,
        tool_choice=_tool_choice(body.get("tool_choice")),
        response_format=response_format,
        sampling=sampling,
        parallel_tool_calls=body.get("parallel_tool_calls"),
        effort=effort,
        thinking_enabled=None,
        max_tokens_requested=body.get("max_output_tokens"),
        custom_tool_names=custom_names,
        namespaces=namespaces,
        summary_enabled=summary_enabled,
    )


# --------------------------------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------------------------------


def _error_code(exc: Exception) -> str:
    if isinstance(exc, chat.ContextOverflow):
        return "context_length_exceeded"
    status = getattr(exc, "status", 500)
    if status in (503, 529):
        return "server_is_overloaded"
    if status == 429:
        return "rate_limit_exceeded"
    return "server_error"


# --------------------------------------------------------------------------------------------------
# Streaming SSE writer
# --------------------------------------------------------------------------------------------------


async def stream_sse(
    session: aiohttp.ClientSession,
    engine_url: str,
    body: dict[str, Any],
    *,
    model_id: str,
    parsed: ir.ParsedRequest,
    keepalive: float = 10.0,
) -> AsyncIterator[bytes]:
    resp_id = _rand("resp_")
    created = int(time.time())
    seq = 0

    def emit(event_type: str, data: dict[str, Any]) -> bytes:
        nonlocal seq
        data = {"type": event_type, "sequence_number": seq, **data}
        seq += 1
        return _frame(event_type, data)

    def response_obj(status: str, output: list[dict[str, Any]], usage: dict[str, Any] | None) -> dict[str, Any]:
        return {"id": resp_id, "object": "response", "created_at": created, "status": status,
                "model": model_id, "output": output, "usage": usage}

    yield emit("response.created", {"response": response_obj("in_progress", [], None)})
    yield emit("response.in_progress", {"response": response_obj("in_progress", [], None)})

    items_done: list[dict[str, Any]] = []
    next_index = 0

    # Open content item state: ("reasoning"|"message", output_index, item_id, text_accum).
    open_item: dict[str, Any] | None = None
    # Buffered tool calls, in arrival order: {index: {"id","name","args"}}.
    tool_buf: dict[int, dict[str, Any]] = {}
    tool_order: list[int] = []
    finish_reason: str | None = None
    usage = ir.Usage()

    def close_content() -> list[bytes]:
        nonlocal open_item
        if open_item is None:
            return []
        frames: list[bytes] = []
        kind = open_item["kind"]
        idx = open_item["index"]
        item_id = open_item["id"]
        text = open_item["text"]
        if kind == "reasoning":
            frames.append(emit("response.reasoning_text.done",
                               {"item_id": item_id, "output_index": idx, "content_index": 0, "text": text}))
            item = {"id": item_id, "type": "reasoning", "summary": [],
                    "content": [{"type": "reasoning_text", "text": text}], "encrypted_content": None}
        else:
            frames.append(emit("response.output_text.done",
                               {"item_id": item_id, "output_index": idx, "content_index": 0, "text": text, "logprobs": []}))
            frames.append(emit("response.content_part.done",
                               {"item_id": item_id, "output_index": idx, "content_index": 0,
                                "part": {"type": "output_text", "text": text, "annotations": []}}))
            item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}
        frames.append(emit("response.output_item.done", {"output_index": idx, "item": item}))
        items_done.append(item)
        open_item = None
        return frames

    def ensure_content(kind: str) -> tuple[dict[str, Any], list[bytes]]:
        nonlocal open_item, next_index
        if open_item is not None and open_item["kind"] == kind:
            return open_item, []
        frames = close_content()
        idx = next_index
        next_index += 1
        if kind == "reasoning":
            item_id = _rand("rs_")
            open_item = {"kind": kind, "index": idx, "id": item_id, "text": ""}
            frames.append(emit("response.output_item.added", {"output_index": idx, "item": {
                "id": item_id, "type": "reasoning", "summary": [], "content": None, "encrypted_content": None}}))
        else:
            item_id = _rand("msg_")
            open_item = {"kind": kind, "index": idx, "id": item_id, "text": ""}
            frames.append(emit("response.output_item.added", {"output_index": idx, "item": {
                "id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}}))
            frames.append(emit("response.content_part.added", {"item_id": item_id, "output_index": idx,
                "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}))
        return open_item, frames

    source = chat.stream_events(session, engine_url, body)
    try:
        async for item in chat.stream_with_keepalive(source, keepalive):
            if item is chat.KEEPALIVE:
                yield emit("response.in_progress", {"response": response_obj("in_progress", [], None)})
                continue
            if isinstance(item, ir.ReasoningDelta):
                state, frames = ensure_content("reasoning")
                for f in frames:
                    yield f
                state["text"] += item.text
                yield emit("response.reasoning_text.delta",
                           {"item_id": state["id"], "output_index": state["index"], "content_index": 0, "delta": item.text})
            elif isinstance(item, ir.TextDelta):
                state, frames = ensure_content("message")
                for f in frames:
                    yield f
                state["text"] += item.text
                yield emit("response.output_text.delta",
                           {"item_id": state["id"], "output_index": state["index"], "content_index": 0,
                            "delta": item.text, "logprobs": []})
            elif isinstance(item, ir.ToolCallStart):
                if item.index not in tool_buf:
                    tool_order.append(item.index)
                    tool_buf[item.index] = {"id": item.id, "name": item.name, "args": ""}
            elif isinstance(item, ir.ToolArgsDelta):
                if item.index in tool_buf:
                    tool_buf[item.index]["args"] += item.delta
            elif isinstance(item, ir.Finish):
                finish_reason = item.reason
            elif isinstance(item, ir.UsageEvent):
                usage = item.usage
    except Exception as exc:  # noqa: BLE001 - surface upstream failures as response.failed
        for f in close_content():
            yield f
        yield emit("response.failed", {"response": {
            "id": resp_id, "object": "response", "created_at": created, "status": "failed",
            "model": model_id, "output": items_done,
            "error": {"code": _error_code(exc), "message": str(exc)}}})
        return

    for f in close_content():
        yield f

    # Emit buffered tool calls as whole items (Codex consumes output_item.done).
    for index in tool_order:
        buf = tool_buf[index]
        idx = next_index
        next_index += 1
        flat = buf["name"]
        call_id = buf["id"]
        args = buf["args"]
        if flat in parsed.custom_tool_names:
            yield emit("response.output_item.added", {"output_index": idx, "item": {
                "id": "ctc_" + call_id, "type": "custom_tool_call", "status": "in_progress",
                "call_id": call_id, "name": flat, "input": ""}})
            try:
                decoded = json.loads(args).get("input", "") if args else ""
            except (ValueError, AttributeError):
                decoded = args
            yield emit("response.custom_tool_call_input.delta",
                       {"item_id": "ctc_" + call_id, "output_index": idx, "delta": decoded})
            yield emit("response.custom_tool_call_input.done",
                       {"item_id": "ctc_" + call_id, "output_index": idx, "input": decoded})
            item = {"id": "ctc_" + call_id, "type": "custom_tool_call", "status": "completed",
                    "call_id": call_id, "name": flat, "input": decoded}
        else:
            namespace = parsed.namespaces.get(flat)
            name = flat[len(namespace) + 2:] if namespace else flat
            added = {"id": "fc_" + call_id, "type": "function_call", "status": "in_progress",
                     "call_id": call_id, "name": name, "arguments": ""}
            if namespace:
                added["namespace"] = namespace
            yield emit("response.output_item.added", {"output_index": idx, "item": added})
            yield emit("response.function_call_arguments.delta",
                       {"item_id": "fc_" + call_id, "output_index": idx, "delta": args})
            yield emit("response.function_call_arguments.done",
                       {"item_id": "fc_" + call_id, "output_index": idx, "arguments": args})
            item = {"id": "fc_" + call_id, "type": "function_call", "status": "completed",
                    "call_id": call_id, "name": name, "arguments": args}
            if namespace:
                item["namespace"] = namespace
        yield emit("response.output_item.done", {"output_index": idx, "item": item})
        items_done.append(item)

    usage_obj = {
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {"cached_tokens": usage.cached_tokens, "cache_write_tokens": 0},
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        "total_tokens": usage.total_tokens,
    }
    if finish_reason == "length":
        yield emit("response.incomplete", {"response": {
            **response_obj("incomplete", items_done, usage_obj),
            "incomplete_details": {"reason": "max_output_tokens"}}})
    elif finish_reason in ("content_filter", "refusal"):
        yield emit("response.incomplete", {"response": {
            **response_obj("incomplete", items_done, usage_obj),
            "incomplete_details": {"reason": "content_filter"}}})
    else:
        yield emit("response.completed", {"response": {
            **response_obj("completed", items_done, usage_obj), "end_turn": True}})
