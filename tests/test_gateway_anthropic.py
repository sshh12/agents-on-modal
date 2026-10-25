"""Anthropic Messages gateway tests (streaming + non-streaming, tools, errors, keepalive)."""

from __future__ import annotations

from test_gateway_fakes import (
    c_finish, c_reason, c_text, c_tool_args, c_tool_start, c_usage,
    gateway, http_error, model, parse_sse, streaming,
)


def _types(events):
    return [data.get("type") for _event, data in events if data]


async def test_messages_stream_text_has_event_lines_and_full_sequence():
    chunks = [c_text("Hello"), c_text(" world"), c_finish("stop"), c_usage()]
    async with gateway(model(), streaming(chunks)) as gw:
        resp = await gw.post("/v1/messages", json={"model": "x", "stream": True,
                                                   "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        events = parse_sse(await resp.text())
    # Claude Code requires event: lines -> every frame must carry one.
    assert all(event is not None for event, _ in events)
    types = _types(events)
    assert types[0] == "message_start"
    assert "content_block_start" in types and "content_block_delta" in types and "content_block_stop" in types
    assert "message_delta" in types and types[-1] == "message_stop"
    # Text reassembles and stop_reason is end_turn.
    text = "".join(d["delta"]["text"] for e, d in events
                    if d and d.get("type") == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert text == "Hello world"
    delta = next(d for e, d in events if d and d.get("type") == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"] == {"input_tokens": 90, "cache_read_input_tokens": 10, "output_tokens": 20}


async def test_messages_stream_thinking_then_text():
    chunks = [c_reason("let me think"), c_text("answer"), c_finish("stop"), c_usage()]
    async with gateway(model(), streaming(chunks)) as gw:
        resp = await gw.post("/v1/messages", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        events = parse_sse(await resp.text())
    starts = [d["content_block"]["type"] for e, d in events if d and d.get("type") == "content_block_start"]
    assert starts == ["thinking", "text"]


async def test_messages_stream_tool_use_roundtrip():
    chunks = [c_tool_start(0, "toolu_1", "Bash", '{"cmd":'), c_tool_args(0, '"ls"}'), c_finish("tool_calls"), c_usage()]
    capture: list = []
    body = {
        "stream": True,
        "tools": [{"name": "Bash", "input_schema": {"$schema": "x", "type": "object",
                                                     "properties": {"cmd": {"type": "string"}}}}],
        "messages": [{"role": "user", "content": "run ls"}],
    }
    async with gateway(model(), streaming(chunks, capture=capture)) as gw:
        resp = await gw.post("/v1/messages", json=body)
        events = parse_sse(await resp.text())
    # Request side: the tool schema reached the engine as a function with $schema stripped.
    sent_tool = capture[0]["tools"][0]
    assert sent_tool["type"] == "function" and sent_tool["function"]["name"] == "Bash"
    assert "$schema" not in sent_tool["function"]["parameters"]
    # Response side: a tool_use block with reassembled args and stop_reason tool_use.
    block = next(d["content_block"] for e, d in events
                 if d and d.get("type") == "content_block_start" and d["content_block"]["type"] == "tool_use")
    assert block["id"] == "toolu_1" and block["name"] == "Bash"
    args = "".join(d["delta"]["partial_json"] for e, d in events
                   if d and d.get("type") == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
    assert args == '{"cmd":"ls"}'
    delta = next(d for e, d in events if d and d.get("type") == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"


async def test_messages_parallel_tool_calls_flushed_sequentially():
    chunks = [
        c_tool_start(0, "c0", "Read", '{"p":"a"}'),
        c_tool_start(1, "c1", "Read", '{"p":"b"}'),
        c_finish("tool_calls"), c_usage(),
    ]
    async with gateway(model(), streaming(chunks)) as gw:
        resp = await gw.post("/v1/messages", json={"stream": True,
                                                   "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
                                                   "messages": [{"role": "user", "content": "x"}]})
        events = parse_sse(await resp.text())
    tool_blocks = [d["content_block"] for e, d in events
                   if d and d.get("type") == "content_block_start" and d["content_block"]["type"] == "tool_use"]
    assert [b["id"] for b in tool_blocks] == ["c0", "c1"]
    # Blocks never interleave: each start is followed by its stop before the next start.
    seq = [d["type"] for e, d in events if d and d["type"] in ("content_block_start", "content_block_stop")]
    assert seq == ["content_block_start", "content_block_stop", "content_block_start", "content_block_stop"]


async def test_messages_request_translation_history():
    capture: list = []
    body = {
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc=1"},
                   {"type": "text", "text": "You are helpful."}],
        "stream": True,
        "messages": [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "think", "signature": ""},
                {"type": "tool_use", "id": "toolu_9", "name": "Read", "input": {"file_path": "a.txt"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "data"}]},
        ],
    }
    async with gateway(model(), streaming([c_text("ok"), c_finish(), c_usage()], capture=capture)) as gw:
        await (await gw.post("/v1/messages", json=body)).text()
    msgs = capture[0]["messages"]
    # Billing header stripped; remaining system text kept.
    assert msgs[0] == {"role": "system", "content": "You are helpful."}
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "think"
    assert assistant["tool_calls"][0]["id"] == "toolu_9"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"file_path":"a.txt"}'
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "toolu_9" and tool_msg["content"] == "data"


async def test_messages_non_streaming():
    chunks = [c_reason("r"), c_text("final"), c_finish("stop"), c_usage(prompt=50, completion=7, cached=5)]
    async with gateway(model(), streaming(chunks)) as gw:
        resp = await gw.post("/v1/messages", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status == 200
        data = await resp.json()
    assert data["type"] == "message" and data["role"] == "assistant"
    kinds = [b["type"] for b in data["content"]]
    assert kinds == ["thinking", "text"]
    assert data["stop_reason"] == "end_turn"
    assert data["usage"]["input_tokens"] == 45 and data["usage"]["cache_read_input_tokens"] == 5


async def test_messages_max_tokens_clamped():
    capture: list = []
    small = model(context_window=1000, max_output_tokens=400)
    async with gateway(small, streaming([c_text("ok"), c_finish(), c_usage()], capture=capture)) as gw:
        await (await gw.post("/v1/messages", json={"stream": True, "max_tokens": 999999,
                                                   "messages": [{"role": "user", "content": "hi"}]})).text()
    assert capture[0]["max_tokens"] <= 400


async def test_messages_reasoning_mapped_into_chat_body():
    capture: list = []
    async with gateway(model(), streaming([c_text("ok"), c_finish(), c_usage()], capture=capture)) as gw:
        await (await gw.post("/v1/messages", json={"stream": True, "output_config": {"effort": "high"},
                                                   "thinking": {"type": "adaptive"},
                                                   "messages": [{"role": "user", "content": "hi"}]})).text()
    body = capture[0]
    assert body["reasoning_effort"] == "xhigh"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["model"] == "qwen3.8-27b"  # client model ignored


async def test_messages_context_overflow_stream_and_nonstream():
    msg = "The input (250000 tokens) is longer than the model's context length (131072 tokens)."
    body = {"object": "error", "message": msg, "type": "BadRequestError", "code": 400}
    import json as _json
    async with gateway(model(), http_error(400, _json.dumps(body))) as gw:
        # Streaming: an error frame with the Claude-parseable wording.
        resp = await gw.post("/v1/messages", json={"stream": True, "messages": [{"role": "user", "content": "x"}]})
        events = parse_sse(await resp.text())
        err = next(d for e, d in events if d and d.get("type") == "error")
        assert err["error"]["type"] == "invalid_request_error"
        assert err["error"]["message"] == "prompt is too long: 250000 tokens > 131072 maximum"
        # Non-streaming: a real HTTP 400 with the same body.
        resp2 = await gw.post("/v1/messages", json={"messages": [{"role": "user", "content": "x"}]})
        assert resp2.status == 400
        data = await resp2.json()
        assert "prompt is too long" in data["error"]["message"]


async def test_messages_keepalive_pings_during_prefill():
    chunks = [c_text("late"), c_finish(), c_usage()]
    async with gateway(model(), streaming(chunks, delay=0.25), keepalive=0.05) as gw:
        resp = await gw.post("/v1/messages", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        events = parse_sse(await resp.text())
    pings = [d for e, d in events if d and d.get("type") == "ping"]
    assert len(pings) >= 2  # the initial ping plus keepalive pings during the delay


async def test_count_tokens():
    async with gateway(model(), streaming([])) as gw:
        resp = await gw.post("/v1/messages/count_tokens",
                             json={"messages": [{"role": "user", "content": "hello there"}]})
        data = await resp.json()
    assert isinstance(data["input_tokens"], int) and data["input_tokens"] > 0
