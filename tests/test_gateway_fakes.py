"""Shared helpers for the gateway tests: a fake chat-completions engine + SSE utilities.

This file defines no ``test_*`` functions; it is imported by the real gateway test modules
(``tests/`` is on ``sys.path`` in pytest's default import mode).
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Callable

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def model(**overrides: Any) -> dict[str, Any]:
    """A ``Model.to_json()``-shaped dict (Qwen3.8 reasoning table by default)."""

    spec = {
        "name": "qwen3.8-27b",
        "context_window": 131072,
        "max_output_tokens": 32000,
        "reasoning": {
            "default": "high",
            "efforts": {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"},
            "kwarg": "enable_thinking",
        },
        "sampling": {},
    }
    spec.update(overrides)
    return spec


# ---- chat chunk builders (SGLang/llama.cpp shaped) ---------------------------------------------


def c_text(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def c_reason(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}]}


def c_tool_start(index: int, call_id: str, name: str, args: str = "") -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]},
        "finish_reason": None}]}


def c_tool_args(index: int, args: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": index, "id": None, "function": {"name": None, "arguments": args}}]}, "finish_reason": None}]}


def c_finish(reason: str = "stop", matched: str | None = None) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason, "matched_stop": matched}]}


def c_usage(prompt: int = 100, completion: int = 20, cached: int = 10, reasoning: int = 5) -> dict[str, Any]:
    return {"choices": [], "usage": {
        "prompt_tokens": prompt, "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": prompt + completion}}


# ---- fake engine handlers ----------------------------------------------------------------------


def streaming(chunks: list[dict] | Callable[[dict], list[dict]], *, capture: list | None = None,
              delay: float = 0.0) -> Callable:
    """A handler that replies to the gateway's health probe and streams ``chunks`` for a turn."""

    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if capture is not None:
            capture.append(body)
        if not body.get("stream"):  # the /health 1-token probe
            return web.json_response({"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        if delay:
            await asyncio.sleep(delay)
        items = chunks(body) if callable(chunks) else chunks
        for chunk in items:
            await resp.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    return handler


def http_error(status: int, body_text: str) -> Callable:
    async def handler(request: web.Request) -> web.Response:
        await request.json()
        return web.Response(text=body_text, status=status, content_type="application/json")

    return handler


def mid_stream_error(chunks_before: list[dict], error: dict) -> Callable:
    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if not body.get("stream"):
            return web.json_response({"choices": [{"message": {"content": "ok"}}]})
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for chunk in chunks_before:
            await resp.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        await resp.write(("data: " + json.dumps({"error": error}) + "\n\n").encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    return handler


def health_toggle(state: dict, chunks: list[dict] | None = None) -> Callable:
    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if not body.get("stream"):
            if state.get("up"):
                return web.json_response({"choices": [{"message": {"content": "ok"}}]})
            return web.json_response({"error": {"message": "loading"}}, status=503)
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for chunk in chunks or [c_text("hi"), c_finish(), c_usage()]:
            await resp.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    return handler


# ---- harness ------------------------------------------------------------------------------------


@asynccontextmanager
async def gateway(model_spec: dict, handler: Callable, *, keepalive: float | None = None):
    """Start the fake engine + the gateway and yield a ``TestClient`` for the gateway."""

    from agents_on_modal.gateway import build_app

    engine_app = web.Application()
    engine_app.router.add_post("/v1/chat/completions", handler)
    engine = TestServer(engine_app)
    await engine.start_server()
    engine_url = str(engine.make_url("")).rstrip("/")

    prev = os.environ.get("AOM_GATEWAY_KEEPALIVE")
    if keepalive is not None:
        os.environ["AOM_GATEWAY_KEEPALIVE"] = str(keepalive)
    try:
        gw_app = build_app(model_spec, engine_url)
    finally:
        if keepalive is not None:
            if prev is None:
                os.environ.pop("AOM_GATEWAY_KEEPALIVE", None)
            else:
                os.environ["AOM_GATEWAY_KEEPALIVE"] = prev

    client = TestClient(TestServer(gw_app))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()
        await engine.close()


def parse_sse(text: str) -> list[tuple[str | None, Any]]:
    """Parse an SSE body into ``(event, data)`` pairs."""

    events: list[tuple[str | None, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if data == "[DONE]":
            continue
        events.append((event, json.loads(data) if data else None))
    return events
