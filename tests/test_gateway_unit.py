"""Unit tests for the gateway's pure helpers: reasoning mapping, IR rendering, clamping."""

from __future__ import annotations

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
