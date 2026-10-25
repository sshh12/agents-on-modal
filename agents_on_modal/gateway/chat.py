"""The gateway's one upstream client: the engine's streaming ``/v1/chat/completions``.

Everything (streaming Anthropic, streaming Responses, their non-streaming aggregations, and the
chat pass-through) ultimately flows through :func:`stream_events`, which POSTs ``stream=true`` and
turns chat SSE chunks into the IR stream events in :mod:`.ir`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator

import aiohttp

from ..config import AomError
from . import ir

# Sentinel yielded by :func:`stream_with_keepalive` when the upstream is idle.
KEEPALIVE = object()


class UpstreamError(AomError):
    """An error returned by the engine (HTTP non-200 or a mid-stream error frame)."""

    def __init__(self, message: str, *, status: int = 502, code: str = "api_error") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class ContextOverflow(UpstreamError):
    """The prompt (plus requested completion) exceeds the model context window."""

    def __init__(self, message: str, *, prompt_tokens: int | None, max_tokens: int | None) -> None:
        super().__init__(message, status=400, code="context_length_exceeded")
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens


# SGLang / llama.cpp context-overflow message shapes (research C.3).
_CTX_PATTERNS = (
    re.compile(r"input \((\d+) tokens\) is longer than the model's context length \((\d+) tokens\)"),
    re.compile(
        r"maximum context length of (?P<ctx>\d+) tokens\. You requested a total of (?P<total>\d+) "
        r"tokens: (?P<prompt>\d+) tokens from the input messages"
    ),
    re.compile(r"n_prompt_tokens\D+(?P<prompt>\d+).*n_ctx\D+(?P<ctx>\d+)", re.DOTALL),
)


def classify_error(message: str, *, status: int = 502, type_hint: str = "") -> UpstreamError:
    """Turn an engine error message into an :class:`UpstreamError`, detecting context overflow."""

    text = message or ""
    low = text.lower()
    if "exceed_context_size" in type_hint or "context length" in low or "context size" in low or (
        "longer than" in low and "context" in low
    ):
        prompt = ctx = None
        for pattern in _CTX_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groupdict()
                if "prompt" in groups:
                    prompt = int(groups["prompt"])
                    ctx = int(groups.get("ctx") or 0) or None
                else:
                    prompt = int(match.group(1))
                    ctx = int(match.group(2))
                break
        return ContextOverflow(text, prompt_tokens=prompt, max_tokens=ctx)
    return UpstreamError(text or "upstream error", status=status, code=type_hint or "api_error")


def _error_from_body(status: int, body: str) -> UpstreamError:
    """Classify an HTTP error body (SGLang top-level or OpenAI/llama.cpp ``{"error":{...}}``)."""

    message = body
    type_hint = ""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                message = err.get("message", body)
                type_hint = str(err.get("type") or err.get("code") or "")
            elif "message" in parsed:  # SGLang top-level {"object":"error","message":...}
                message = parsed.get("message", body)
                type_hint = str(parsed.get("type") or "")
    except (ValueError, TypeError):
        pass
    return classify_error(message, status=status, type_hint=type_hint)


async def _iter_sse_data(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
    """Yield the JSON payload of each ``data:`` line, stopping at ``[DONE]``."""

    buffer = b""
    async for raw in response.content:
        buffer += raw
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            text = line.strip()
            if not text or text.startswith(b":"):
                continue
            if not text.startswith(b"data:"):
                continue
            payload = text[len(b"data:"):].strip()
            if payload == b"[DONE]":
                return
            yield payload.decode("utf-8", "replace")


async def post_chunks(
    session: aiohttp.ClientSession, engine_url: str, body: dict[str, Any]
) -> AsyncIterator[dict[str, Any]]:
    """POST the chat body and yield parsed chunk dicts; raise :class:`UpstreamError` on failure."""

    url = engine_url.rstrip("/") + "/v1/chat/completions"
    try:
        resp = await session.post(url, json=body)
    except aiohttp.ClientError as exc:
        raise UpstreamError(f"cannot reach engine: {exc}", status=503, code="overloaded_error") from exc
    async with resp:
        if resp.status != 200:
            raise _error_from_body(resp.status, await resp.text())
        async for payload in _iter_sse_data(resp):
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if isinstance(chunk, dict) and chunk.get("error"):
                err = chunk["error"]
                if isinstance(err, dict):
                    raise classify_error(err.get("message", ""), type_hint=str(err.get("type") or ""))
                raise classify_error(str(err))
            if isinstance(chunk, dict):
                yield chunk


async def stream_events(
    session: aiohttp.ClientSession, engine_url: str, body: dict[str, Any]
) -> AsyncIterator[ir.StreamEvent]:
    """Translate the upstream chat stream into IR stream events."""

    seen_tool_index: set[int] = set()
    async for chunk in post_chunks(session, engine_url, body):
        choices = chunk.get("choices") or []
        if not choices:
            usage = chunk.get("usage")
            if usage:
                yield _usage_event(usage)
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}

        reasoning = delta.get("reasoning_content")
        if reasoning is None:
            reasoning = delta.get("reasoning")  # vLLM spelling
        if reasoning:
            yield ir.ReasoningDelta(reasoning)

        content = delta.get("content")
        if content:
            yield ir.TextDelta(content)

        for tc in delta.get("tool_calls") or []:
            index = tc.get("index", 0)
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name is not None and index not in seen_tool_index:
                seen_tool_index.add(index)
                yield ir.ToolCallStart(index=index, id=tc.get("id") or f"call_{index}", name=name)
            args = fn.get("arguments")
            if args:
                yield ir.ToolArgsDelta(index=index, delta=args)

        finish = choice.get("finish_reason")
        if finish:
            matched = choice.get("matched_stop")
            yield ir.Finish(reason=finish, stop_matched=matched if isinstance(matched, str) else None)

        usage = chunk.get("usage")
        if usage:
            yield _usage_event(usage)


def _usage_event(usage: dict[str, Any]) -> ir.UsageEvent:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    comp_details = usage.get("completion_tokens_details") or {}
    reasoning = int(comp_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    return ir.UsageEvent(
        ir.Usage(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached, reasoning_tokens=reasoning)
    )


@dataclass
class ToolCallAccum:
    index: int
    id: str
    name: str
    args: str = ""


@dataclass
class Aggregated:
    """The whole non-streamed turn, built by draining :func:`stream_events`."""

    reasoning: str = ""
    text: str = ""
    tool_calls: list[ToolCallAccum] = None  # type: ignore[assignment]
    finish_reason: str | None = None
    stop_matched: str | None = None
    usage: ir.Usage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []
        if self.usage is None:
            self.usage = ir.Usage()


async def aggregate(
    session: aiohttp.ClientSession, engine_url: str, body: dict[str, Any]
) -> Aggregated:
    """Drain the stream into an :class:`Aggregated` result (the non-streaming code path)."""

    result = Aggregated()
    by_index: dict[int, ToolCallAccum] = {}
    async for event in stream_events(session, engine_url, body):
        if isinstance(event, ir.ReasoningDelta):
            result.reasoning += event.text
        elif isinstance(event, ir.TextDelta):
            result.text += event.text
        elif isinstance(event, ir.ToolCallStart):
            accum = ToolCallAccum(index=event.index, id=event.id, name=event.name)
            by_index[event.index] = accum
            result.tool_calls.append(accum)
        elif isinstance(event, ir.ToolArgsDelta):
            if event.index in by_index:
                by_index[event.index].args += event.delta
        elif isinstance(event, ir.Finish):
            result.finish_reason = event.reason
            result.stop_matched = event.stop_matched
        elif isinstance(event, ir.UsageEvent):
            result.usage = event.usage
        elif isinstance(event, ir.ErrorEvent):
            raise UpstreamError(event.message, status=event.status, code=event.code)
    return result


async def stream_with_keepalive(
    source: AsyncIterator[Any], interval: float
) -> AsyncIterator[Any]:
    """Relay ``source`` but yield :data:`KEEPALIVE` whenever it is idle longer than ``interval``.

    A background task pumps the source into a queue so a slow first byte (cold start / prefill)
    produces keepalive ticks rather than a stalled stream.
    """

    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    async def pump() -> None:
        try:
            async for item in source:
                await queue.put(("item", item))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer side
            await queue.put(("error", exc))
        finally:
            await queue.put(("done", done))

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), interval)
            except asyncio.TimeoutError:
                yield KEEPALIVE
                continue
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:
                return
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
