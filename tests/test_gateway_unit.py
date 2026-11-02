"""Unit tests for the gateway's pure helpers: reasoning mapping, IR rendering, clamping."""

from __future__ import annotations

import json

from agents_on_modal.gateway import ir
from agents_on_modal.gateway.reasoning import chat_reasoning_fields

QWEN38 = {"default": "high", "kwarg": "enable_thinking",
          "efforts": {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}}
DEEPSEEK = {"default": "high", "kwarg": "thinking",
            "efforts": {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}}
GPT_OSS = {"default": "medium", "kwarg": None,
           "efforts": {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}}


def test_reasoning_qwen38_high_to_xhigh():
    fields = chat_reasoning_fields("high", QWEN38)
    assert fields == {"reasoning_effort": "xhigh", "chat_template_kwargs": {"enable_thinking": True}}


def test_reasoning_qwen38_medium_and_low():
    assert chat_reasoning_fields("medium", QWEN38)["reasoning_effort"] == "medium"
    assert chat_reasoning_fields("low", QWEN38)["reasoning_effort"] == "low"


def test_reasoning_deepseek_medium_to_high():
    fields = chat_reasoning_fields("medium", DEEPSEEK)
    assert fields == {"reasoning_effort": "high", "chat_template_kwargs": {"thinking": True}}
    assert chat_reasoning_fields("xhigh", DEEPSEEK)["reasoning_effort"] == "max"


def test_reasoning_gpt_oss_no_template_kwarg():
    fields = chat_reasoning_fields("high", GPT_OSS)
    assert fields == {"reasoning_effort": "high"}
    assert "chat_template_kwargs" not in fields


def test_reasoning_default_when_effort_absent():
    # ``ultra`` normalises to ``max``; ``None`` falls back to the model default.
    assert chat_reasoning_fields("ultra", QWEN38)["reasoning_effort"] == "xhigh"
    assert chat_reasoning_fields(None, DEEPSEEK)["reasoning_effort"] == "high"


def test_reasoning_disabled_with_kwarg():
    assert chat_reasoning_fields("high", QWEN38, enabled=False) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_reasoning_disabled_without_kwarg():
    assert chat_reasoning_fields("high", GPT_OSS, enabled=False) == {"reasoning_effort": "none"}


def test_clamp_prefers_headroom():
    # Context 1000, prompt estimate 200, margin 64 -> headroom 736 beats a huge request.
    assert ir.clamp_max_tokens(999999, context_window=1000, prompt_estimate=200,
                               model_max_output=500, default=500) == 500
    assert ir.clamp_max_tokens(999999, context_window=1000, prompt_estimate=900,
                               model_max_output=500, default=500) == 1000 - 900 - 64


def test_clamp_default_and_floor():
    assert ir.clamp_max_tokens(None, context_window=131072, prompt_estimate=10,
                               model_max_output=32000, default=32000) == 32000
    assert ir.clamp_max_tokens(100, context_window=50, prompt_estimate=100,
                               model_max_output=32000, default=32000) == 1


def test_to_chat_messages_tool_result_before_text_and_system_reminder():
    msgs = [
        ir.Message("user", [ir.TextPart("do it")]),
        ir.Message("assistant", [ir.ReasoningPart("hmm"), ir.TextPart("ok"),
                                 ir.ToolCallPart(ir.ToolCall(id="t1", name="Bash", args_json='{"cmd":"ls"}'))]),
        ir.Message("user", [ir.ToolResultPart(id="t1", content="output")]),
        ir.Message("system", [ir.TextPart("env note")]),
    ]
    chat = ir.to_chat_messages("SYS", msgs)
    assert chat[0] == {"role": "system", "content": "SYS"}
    assert chat[1] == {"role": "user", "content": "do it"}
    assistant = chat[2]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == "hmm"
    assert assistant["tool_calls"][0]["function"] == {"name": "Bash", "arguments": '{"cmd":"ls"}'}
    # The tool result becomes a tool message; the trailing system message is folded into it.
    tool_msg = chat[3]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "t1" and tool_msg["content"] == "output"
    assert chat[-1]["role"] == "user" and "<system-reminder>" in chat[-1]["content"]


def test_tool_result_error_prefix():
    msgs = [ir.Message("user", [ir.ToolResultPart(id="x", content="boom", is_error=True)])]
    chat = ir.to_chat_messages("", msgs)
    assert chat[0]["content"] == "Error: boom"


def test_usage_cache_semantics():
    usage = ir.Usage(prompt_tokens=100, completion_tokens=20, cached_tokens=30)
    assert usage.input_tokens_ex_cache == 70  # Anthropic excludes cache reads
    assert usage.total_tokens == 120


async def test_post_chunks_retries_a_dropped_engine_connection(monkeypatch):
    # A stale pooled keep-alive connection fails the POST before any response; a retry fixes it
    # (surfaced as a 503 instead, Codex fails the whole turn with "model is at capacity").
    import aiohttp

    from agents_on_modal.gateway import chat

    class Resp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    calls = {"n": 0}

    class Session:
        async def post(self, url, json):
            calls["n"] += 1
            if calls["n"] == 1:
                raise aiohttp.ServerDisconnectedError()
            return Resp()

    async def one_chunk(_resp):
        yield '{"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]}'

    monkeypatch.setattr(chat, "_iter_sse_data", one_chunk)
    monkeypatch.setattr(chat.asyncio, "sleep", lambda *_: _noop())
    got = [c async for c in chat.post_chunks(Session(), "http://engine", {})]
    assert calls["n"] == 2
    assert got[0]["choices"][0]["delta"]["content"] == "hi"


async def _noop():
    return None


def test_dp_size_of_reads_sglang_dp_flags():
    assert ir.dp_size_of(None) == 1
    assert ir.dp_size_of(["--enable-metrics", "--ep-size=4"]) == 1
    assert ir.dp_size_of(["--enable-dp-attention", "--dp-size=4"]) == 4
    assert ir.dp_size_of(["--dp=2"]) == 2
    assert ir.dp_size_of(["--attn-dp-size=4"]) == 4


def test_dp_rank_is_stable_per_conversation_and_spreads_conversations():
    assert ir.dp_rank_for("session-a", 4) == ir.dp_rank_for("session-a", 4)
    ranks = {ir.dp_rank_for(f"session-{i}", 4) for i in range(64)}
    assert ranks == {0, 1, 2, 3}
    # Without a client key, later turns of a conversation still hash like its first turn.
    first = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task 7"}]
    later = first + [{"role": "assistant", "content": "ok"}, {"role": "tool", "tool_call_id": "c", "content": "x"},
                     {"role": "user", "content": "more"}]
    assert ir.affinity_key(first) == ir.affinity_key(later)


def test_history_args_keep_valid_objects_and_wrap_anything_else():
    assert ir.history_args('{"cmd":"ls"}') == '{"cmd":"ls"}'
    assert ir.history_args("") == ""
    truncated = '{"cmd": "python -m pytest -q && echo do'
    assert json.loads(ir.history_args(truncated)) == {"_invalid_arguments": truncated}
    assert json.loads(ir.history_args('["a"]')) == {"_invalid_arguments": '["a"]'}


def test_repetition_guard_flags_short_period_loops_only():
    guard = ir.RepetitionGuard(window=1000)
    lines = [f"def f{i}(w, h):\n    return w * h + {i}  # the area plus {i}, see the docstring\n" for i in range(60)]
    for line in lines:  # long, ordinary code: never periodic over 1000 chars
        assert guard.feed(line) is None
    loop = ir.RepetitionGuard(window=1000)
    found = None
    for _ in range(600):
        found = loop.feed("The") or found
    assert found == 3
    assert ir.RepetitionGuard(window=0).feed("x" * 5000) is None  # disabled


async def test_stream_cut_on_runaway_repetition_finishes_as_length(monkeypatch):
    from agents_on_modal.gateway import chat

    async def fake_chunks(_session, _url, _body):
        yield {"choices": [{"delta": {"content": "Starting. "}}]}
        for _ in range(2000):
            yield {"choices": [{"delta": {"reasoning_content": "Fail"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    monkeypatch.setattr(chat, "post_chunks", fake_chunks)
    events = [e async for e in chat.stream_events(None, "http://engine", {})]
    finishes = [e for e in events if isinstance(e, ir.Finish)]
    assert [f.reason for f in finishes] == ["length"]
    assert sum(1 for e in events if isinstance(e, ir.ReasoningDelta)) < 400  # cut long before the end
