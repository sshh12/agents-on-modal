"""Server-level gateway tests: /v1/models, /health, HEAD, chat pass-through."""

from __future__ import annotations

from test_gateway_fakes import (
    c_finish, c_text, c_usage, gateway, health_toggle, model, parse_sse, streaming,
)


async def test_models_advertises_served_model():
    async with gateway(model(), streaming([])) as gw:
        resp = await gw.get("/v1/models")
        data = await resp.json()
    assert data["data"][0]["id"] == "qwen3.8-27b"
    assert data["data"][0]["max_model_len"] == 131072


async def test_head_root_ok():
    async with gateway(model(), streaming([])) as gw:
        resp = await gw.request("HEAD", "/")
        assert resp.status == 200


async def test_health_503_then_200():
    state = {"up": False}
    async with gateway(model(), health_toggle(state)) as gw:
        resp = await gw.get("/health")
        assert resp.status == 503  # engine not generating yet
        state["up"] = True
        resp2 = await gw.get("/health")
        assert resp2.status == 200
        # Once healthy, it stays healthy (cached) without re-probing.
        state["up"] = False
        resp3 = await gw.get("/health")
        assert resp3.status == 200


async def test_chat_passthrough_streams_raw_and_overrides_model():
    capture: list = []
    chunks = [c_text("raw"), c_finish(), c_usage()]
    async with gateway(model(), streaming(chunks, capture=capture)) as gw:
        resp = await gw.post("/v1/chat/completions", json={
            "model": "client-chose-this", "stream": True, "reasoning_effort": "high", "max_tokens": 999999,
            "messages": [{"role": "user", "content": "hi"}]})
        text = await resp.text()
    # Pass-through: the raw chat chunks come straight back (no Anthropic/Responses framing).
    datas = [d for _e, d in parse_sse(text) if d]
    assert any(d.get("choices") and d["choices"][0].get("delta", {}).get("content") == "raw" for d in datas)
    assert not any("message_start" == d.get("type") for d in datas)  # not translated
    sent = capture[0]
    assert sent["model"] == "qwen3.8-27b"            # model overridden
    assert sent["reasoning_effort"] == "xhigh"        # reasoning mapped
    assert sent["chat_template_kwargs"] == {"enable_thinking": True}
    assert sent["max_tokens"] <= 32000                # clamped


async def test_chat_passthrough_non_stream_returns_json():
    async def handler(request):
        from aiohttp import web
        body = await request.json()
        assert body["model"] == "qwen3.8-27b"
        return web.json_response({"id": "c1", "object": "chat.completion",
                                  "choices": [{"message": {"content": "hey"}}]})

    async with gateway(model(), handler) as gw:
        resp = await gw.post("/v1/chat/completions", json={
            "model": "x", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status == 200
        data = await resp.json()
    assert data["choices"][0]["message"]["content"] == "hey"
