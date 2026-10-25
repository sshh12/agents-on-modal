"""OpenAI Responses gateway tests (Codex): output items, custom + namespace tools, errors."""

from __future__ import annotations

import json

from test_gateway_fakes import (
    c_finish, c_reason, c_text, c_tool_args, c_tool_start, c_usage,
    gateway, http_error, mid_stream_error, model, parse_sse, streaming,
)


def _events_of(data_list, type_):
    return [d for d in data_list if d and d.get("type") == type_]


async def test_responses_stream_completed_with_id_and_usage():
    chunks = [c_reason("thinking"), c_text("OK done"), c_finish("stop"), c_usage(prompt=100, completion=20, cached=10, reasoning=5)]
    async with gateway(model(), streaming(chunks)) as gw:
        resp = await gw.post("/v1/responses", json={"model": "x", "stream": True,
                                                    "input": [{"type": "message", "role": "user",
                                                               "content": [{"type": "input_text", "text": "hi"}]}]})
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        events = [d for _e, d in parse_sse(await resp.text())]
    assert events[0]["type"] == "response.created" and events[0]["response"]["id"]
    # Codex rebuilds history from output_item.done + response.completed.
    done_items = [d["item"] for d in _events_of(events, "response.output_item.done")]
    kinds = [it["type"] for it in done_items]
    assert "reasoning" in kinds and "message" in kinds
    reasoning_item = next(it for it in done_items if it["type"] == "reasoning")
    assert reasoning_item["summary"] == [] and reasoning_item["content"][0]["type"] == "reasoning_text"
    completed = _events_of(events, "response.completed")[0]["response"]
    assert completed["id"] == events[0]["response"]["id"]
    usage = completed["usage"]
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 20 and usage["total_tokens"] == 120
    assert usage["input_tokens_details"]["cached_tokens"] == 10
    assert usage["output_tokens_details"]["reasoning_tokens"] == 5
    assert all(isinstance(usage[k], int) for k in ("input_tokens", "output_tokens", "total_tokens"))
    # sequence_number present and monotonic.
    seqs = [d["sequence_number"] for d in events]
    assert seqs == sorted(seqs)


async def test_responses_function_tool_roundtrip():
    chunks = [c_tool_start(0, "call_a", "exec_command", '{"cmd":'), c_tool_args(0, '"ls"}'), c_finish("tool_calls"), c_usage()]
    capture: list = []
    body = {"stream": True, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run"}]}],
            "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]}
    async with gateway(model(), streaming(chunks, capture=capture)) as gw:
        events = [d for _e, d in parse_sse(await (await gw.post("/v1/responses", json=body)).text())]
    item = next(d["item"] for d in _events_of(events, "response.output_item.done") if d["item"]["type"] == "function_call")
    assert item["call_id"] == "call_a" and item["name"] == "exec_command"
    assert item["arguments"] == '{"cmd":"ls"}' and item["status"] == "completed"


async def test_responses_custom_apply_patch_roundtrip():
    patch = "*** Begin Patch\n*** Add File: x.txt\n+hi\n*** End Patch\n"
    args = json.dumps({"input": patch})
    chunks = [c_tool_start(0, "call_p", "apply_patch", args), c_finish("tool_calls"), c_usage()]
    capture: list = []
    body = {
        "stream": True,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "patch"}]}],
        "tools": [{"type": "custom", "name": "apply_patch", "description": "FREEFORM tool",
                   "format": {"type": "grammar", "syntax": "lark", "definition": "start: hunk+"}}],
    }
    async with gateway(model(), streaming(chunks, capture=capture)) as gw:
        events = [d for _e, d in parse_sse(await (await gw.post("/v1/responses", json=body)).text())]
    # Request: the custom tool became a function with a single string `input` arg + grammar in desc.
    sent = capture[0]["tools"][0]
    assert sent["type"] == "function" and sent["function"]["name"] == "apply_patch"
    assert sent["function"]["parameters"]["required"] == ["input"]
    assert "lark" in sent["function"]["description"]
    # Response: emitted as a custom_tool_call with the unwrapped patch text.
    item = next(d["item"] for d in _events_of(events, "response.output_item.done") if d["item"]["type"] == "custom_tool_call")
    assert item["name"] == "apply_patch" and item["input"] == patch and item["call_id"] == "call_p"
    assert _events_of(events, "response.custom_tool_call_input.done")[0]["input"] == patch


async def test_responses_namespace_flatten_and_restore():
    chunks = [c_tool_start(0, "call_s", "multi_agent_v1__spawn_agent", '{"message":"go"}'), c_finish("tool_calls"), c_usage()]
    capture: list = []
    body = {
        "stream": True,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "spawn"}]}],
        "tools": [{"type": "namespace", "name": "multi_agent_v1", "description": "Spawn sub-agents.",
                   "tools": [{"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}}]}],
    }
    async with gateway(model(), streaming(chunks, capture=capture)) as gw:
        events = [d for _e, d in parse_sse(await (await gw.post("/v1/responses", json=body)).text())]
    # Request: flattened to ns__member.
    assert capture[0]["tools"][0]["function"]["name"] == "multi_agent_v1__spawn_agent"
    # Response: namespace restored on the function_call item.
    item = next(d["item"] for d in _events_of(events, "response.output_item.done") if d["item"]["type"] == "function_call")
    assert item["name"] == "spawn_agent" and item["namespace"] == "multi_agent_v1"


async def test_responses_history_merges_reasoning_into_assistant():
    capture: list = []
    body = {
        "stream": True,
        "instructions": "base",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev rules"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do it"}]},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "because"}]},
            {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "done"},
        ],
    }
    async with gateway(model(), streaming([c_text("ok"), c_finish(), c_usage()], capture=capture)) as gw:
        await (await gw.post("/v1/responses", json=body)).text()
    msgs = capture[0]["messages"]
    # Leading developer folds into the system message.
    assert msgs[0]["role"] == "system" and "base" in msgs[0]["content"] and "dev rules" in msgs[0]["content"]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "because"
    assert assistant["tool_calls"][0]["function"]["name"] == "exec_command"
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1" and tool_msg["content"] == "done"


async def test_responses_length_becomes_incomplete():
    chunks = [c_text("partial"), c_finish("length"), c_usage()]
    async with gateway(model(), streaming(chunks)) as gw:
        events = [d for _e, d in parse_sse(await (await gw.post("/v1/responses", json={
            "stream": True, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]})).text())]
    inc = _events_of(events, "response.incomplete")
    assert inc and inc[0]["response"]["incomplete_details"]["reason"] == "max_output_tokens"


async def test_responses_context_overflow_failed():
    msg = "The input (250000 tokens) is longer than the model's context length (131072 tokens)."
    body = json.dumps({"object": "error", "message": msg, "type": "BadRequestError"})
    async with gateway(model(), http_error(400, body)) as gw:
        events = [d for _e, d in parse_sse(await (await gw.post("/v1/responses", json={
            "stream": True, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]})).text())]
    failed = _events_of(events, "response.failed")
    assert failed and failed[0]["response"]["error"]["code"] == "context_length_exceeded"
    assert failed[0]["response"]["status"] == "failed"


async def test_responses_mid_stream_error_becomes_failed():
    err = {"object": "error", "message": "boom", "type": "InternalServerError"}
    async with gateway(model(), mid_stream_error([c_text("part")], err)) as gw:
        events = [d for _e, d in parse_sse(await (await gw.post("/v1/responses", json={
            "stream": True, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]})).text())]
    assert _events_of(events, "response.failed")


async def test_responses_reasoning_effort_passthrough_of_arbitrary_string():
    capture: list = []
    async with gateway(model(), streaming([c_text("ok"), c_finish(), c_usage()], capture=capture)) as gw:
        await (await gw.post("/v1/responses", json={
            "stream": True, "reasoning": {"effort": "xhigh", "summary": "auto"},
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x"}]}]})).text()
    assert capture[0]["reasoning_effort"] == "xhigh"  # qwen3.8 xhigh -> xhigh
