"""The gateway aiohttp app: Anthropic Messages, OpenAI Responses, and chat pass-through.

Imports only aiohttp + stdlib + the sibling gateway modules (which pull in the modal-free
``agents_on_modal.models`` for effort normalisation), so it loads both inside the GPU container and
in offline unit tests. Build it with :func:`build_app`; serve it with :func:`run`.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any

from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from . import anthropic, chat, responses
from .ir import clamp_max_tokens, estimate_tokens
from .reasoning import chat_reasoning_fields

_SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}


def _request_id() -> str:
    return "req_" + secrets.token_hex(12)


def _apply_reasoning(body: dict[str, Any], fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        if key == "chat_template_kwargs":
            merged = dict(body.get("chat_template_kwargs") or {})
            merged.update(value)
            body["chat_template_kwargs"] = merged
        else:
            body[key] = value


def build_app(model: dict, engine_url: str) -> web.Application:
    """An aiohttp app serving every harness protocol against ``engine_url``'s chat completions.

    ``model`` is :meth:`agents_on_modal.models.Model.to_json`; ``engine_url`` is like
    ``http://127.0.0.1:8001``. The served model id (``model["name"]``) always overrides the client's.
    """

    name = model["name"]
    context_window = int(model["context_window"])
    max_output = int(model["max_output_tokens"])
    reasoning_cfg = model.get("reasoning") or {}
    sampling_defaults = dict(model.get("sampling") or {})
    engine = engine_url.rstrip("/")
    _engine_split = urlsplit(engine)
    engine_host = _engine_split.hostname or "127.0.0.1"
    engine_port = _engine_split.port or (443 if _engine_split.scheme == "https" else 80)
    keepalive = float(os.environ.get("AOM_GATEWAY_KEEPALIVE", "10"))
    #: How long a request will wait for a cold-starting engine before giving up (seconds). The
    #: harnesses use hour-long header/idle timeouts, so holding here keeps a cold start transparent
    #: instead of surfacing the engine's "not up yet" as a hard 503 that aborts the agent's run.
    engine_wait = float(os.environ.get("AOM_GATEWAY_ENGINE_WAIT", "600"))

    app = web.Application(client_max_size=256 * 1024 * 1024)
    # Runtime state lives in a closure dict, not on the (post-startup immutable) app mapping.
    state: dict[str, Any] = {"healthy": False, "session": None}

    async def _probe_engine(session: aiohttp.ClientSession) -> bool:
        """True once the engine can actually serve (SGLang's 1-token generation returns choices)."""

        probe = {"model": name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1, "stream": False}
        try:
            async with session.post(
                engine + "/v1/chat/completions", json=probe, timeout=aiohttp.ClientTimeout(total=15)
            ) as upstream:
                if upstream.status != 200:
                    return False
                data = await upstream.json(content_type=None)
                return bool(data.get("choices"))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return False

    async def _ensure_engine_ready(_session: aiohttp.ClientSession) -> bool:
        """Hold the caller only while the engine's TCP socket is unreachable (or ``engine_wait``).

        A fresh container's gateway binds its port and starts accepting requests well before the
        engine subprocess has loaded the weights (a multi-minute cold start). Without this gate the
        first requests -- e.g. every agent in a just-scaled-up swarm -- would hit a connection
        refused and get a 503 that the harness treats as a fatal error. SGLang binds its HTTP port
        only once it can serve, so a bare TCP connect is the readiness signal: no HTTP request is
        sent (nothing for the proxied request to race or for a genuine engine error to hide behind),
        and once connectable the real request proceeds. Steady state is a single dict lookup.
        """

        if state["healthy"]:
            return True
        loop = asyncio.get_event_loop()
        deadline = loop.time() + engine_wait
        while True:
            try:
                _reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(engine_host, engine_port), timeout=10
                )
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:  # noqa: BLE001 - close is best-effort
                    pass
                state["healthy"] = True
                return True
            except (OSError, asyncio.TimeoutError):
                if loop.time() >= deadline:
                    return False
                await asyncio.sleep(2)

    def build_chat_body(parsed, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": name, "messages": parsed.messages, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        if parsed.tools:
            body["tools"] = parsed.tools
            if parsed.tool_choice is not None:
                body["tool_choice"] = parsed.tool_choice
        if parsed.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parsed.parallel_tool_calls
        if parsed.stop:
            body["stop"] = parsed.stop
        if parsed.response_format:
            body["response_format"] = parsed.response_format
        merged_sampling = {**sampling_defaults, **parsed.sampling}
        body.update(merged_sampling)
        _apply_reasoning(body, chat_reasoning_fields(parsed.effort, reasoning_cfg, enabled=parsed.thinking_enabled))
        prompt_estimate = estimate_tokens(
            json.dumps(parsed.messages, ensure_ascii=False) + json.dumps(parsed.tools, ensure_ascii=False)
        )
        body["max_tokens"] = clamp_max_tokens(
            parsed.max_tokens_requested, context_window=context_window, prompt_estimate=prompt_estimate,
            model_max_output=max_output, default=max_output,
        )
        return body

    # ---- Anthropic Messages -----------------------------------------------------------------
    async def messages_handler(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return web.json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid JSON"}}, status=400)
        session: aiohttp.ClientSession = state["session"]
        if not await _ensure_engine_ready(session):
            return web.json_response(
                {"type": "error", "error": {"type": "overloaded_error", "message": "engine not ready"}},
                status=503, headers={"request-id": _request_id()})
        parsed = anthropic.parse_request(body)
        if not body.get("stream"):
            # Always stream upstream and aggregate: one engine code path for both client modes.
            chat_body = build_chat_body(parsed, stream=True)
            try:
                data = await anthropic.response_json(session, engine, chat_body, model_id=name, stops=parsed.stop)
            except chat.UpstreamError as exc:
                status, payload = anthropic.error_payload(exc, context_window=context_window)
                return web.json_response(payload, status=status, headers={"request-id": _request_id()})
            return web.json_response(data, headers={"request-id": _request_id()})
        chat_body = build_chat_body(parsed, stream=True)
        resp = web.StreamResponse(status=200, headers={**_SSE_HEADERS, "request-id": _request_id()})
        await resp.prepare(request)
        async for chunk in anthropic.stream_sse(
            session, engine, chat_body, model_id=name, stops=parsed.stop, context_window=context_window, keepalive=keepalive
        ):
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    async def count_tokens_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return web.json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid JSON"}}, status=400)
        return web.json_response(anthropic.count_tokens(body), headers={"request-id": _request_id()})

    # ---- OpenAI Responses -------------------------------------------------------------------
    async def responses_handler(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return web.json_response({"error": {"message": "invalid JSON", "type": "invalid_request_error"}}, status=400)
        session = state["session"]
        if not await _ensure_engine_ready(session):
            return web.json_response({"error": {"message": "engine not ready", "type": "api_error"}}, status=503)
        parsed = responses.parse_request(body)
        chat_body = build_chat_body(parsed, stream=True)
        resp = web.StreamResponse(status=200, headers={**_SSE_HEADERS})
        await resp.prepare(request)
        async for chunk in responses.stream_sse(
            session, engine, chat_body, model_id=name, parsed=parsed, keepalive=keepalive
        ):
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    # ---- chat pass-through (reasoning mapping + model override + max_tokens clamp) ----------
    async def chat_handler(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return web.json_response({"error": {"message": "invalid JSON", "type": "invalid_request_error"}}, status=400)
        session = state["session"]
        if not await _ensure_engine_ready(session):
            return web.json_response({"error": {"message": "engine not ready", "type": "api_error"}}, status=503)
        body["model"] = name
        _apply_reasoning(body, chat_reasoning_fields(body.get("reasoning_effort"), reasoning_cfg))
        prompt_estimate = estimate_tokens(json.dumps(body.get("messages") or [], ensure_ascii=False))
        body["max_tokens"] = clamp_max_tokens(
            body.get("max_tokens"), context_window=context_window, prompt_estimate=prompt_estimate,
            model_max_output=max_output, default=max_output,
        )
        stream = bool(body.get("stream"))
        url = engine + "/v1/chat/completions"
        try:
            upstream = await session.post(url, json=body)
        except aiohttp.ClientError as exc:
            return web.json_response({"error": {"message": str(exc), "type": "api_error"}}, status=503)
        if not stream:
            async with upstream:
                payload = await upstream.read()
                return web.Response(body=payload, status=upstream.status, content_type="application/json")
        resp = web.StreamResponse(status=upstream.status, headers={**_SSE_HEADERS})
        await resp.prepare(request)
        async with upstream:
            async for raw in upstream.content.iter_any():
                await resp.write(raw)
        await resp.write_eof()
        return resp

    # ---- discovery, health, probes ----------------------------------------------------------
    async def models_handler(_request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": name, "object": "model", "display_name": name, "max_model_len": context_window,
             "created": 0, "owned_by": "agents-on-modal"}]})

    async def health_handler(request: web.Request) -> web.Response:
        if state["healthy"] or await _probe_engine(state["session"]):
            state["healthy"] = True
            return web.json_response({"status": "ok"})
        return web.json_response({"status": "loading"}, status=503)

    async def head_ok(_request: web.Request) -> web.Response:
        return web.Response(status=200)

    app.router.add_post("/v1/messages", messages_handler)
    app.router.add_post("/v1/messages/count_tokens", count_tokens_handler)
    app.router.add_post("/v1/responses", responses_handler)
    app.router.add_post("/v1/chat/completions", chat_handler)
    app.router.add_get("/v1/models", models_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_route("HEAD", "/", head_ok)
    app.router.add_route("HEAD", "/api/hello", head_ok)

    async def _on_startup(_app: web.Application) -> None:
        # No total timeout: streams and cold starts can run for many minutes.
        state["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=None))

    async def _on_cleanup(_app: web.Application) -> None:
        session = state.get("session")
        if session is not None:
            await session.close()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


async def run(model: dict, engine_url: str, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Build the app and serve it forever."""

    app = build_app(model, engine_url)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    try:
        await asyncio.Event().wait()  # serve until cancelled
    finally:
        await runner.cleanup()
