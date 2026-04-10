"""Unit tests for mlx_anthropic.translation."""

import json
import pytest

from mlx_anthropic.schema import (
    Message, MessagesRequest, TextBlock, ToolUseBlock, ToolResultBlock,
    ToolDefinition, ToolChoiceAuto, ToolChoiceTool,
)
from mlx_anthropic.translation import (
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
    anthropic_to_openai_tool_choice,
    build_openai_request,
    openai_to_anthropic_response,
)


def _make_request(**kwargs) -> MessagesRequest:
    defaults = dict(model="qwen-72b", messages=[], max_tokens=1024)
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


# ---------------------------------------------------------------------------
# anthropic_to_openai_messages
# ---------------------------------------------------------------------------

class TestAnthropicToOpenAIMessages:
    def test_plain_string_message(self):
        req = _make_request(messages=[Message(role="user", content="Hello")])
        msgs = anthropic_to_openai_messages(req)
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_system_string(self):
        req = _make_request(system="Be concise.", messages=[Message(role="user", content="Hi")])
        msgs = anthropic_to_openai_messages(req)
        assert msgs[0] == {"role": "system", "content": "Be concise."}
        assert msgs[1]["role"] == "user"

    def test_system_text_block_list(self):
        req = _make_request(
            system=[TextBlock(text="Part A."), TextBlock(text="Part B.")],
            messages=[],
        )
        msgs = anthropic_to_openai_messages(req)
        assert msgs[0]["content"] == "Part A. Part B."

    def test_text_block_content(self):
        req = _make_request(messages=[
            Message(role="user", content=[TextBlock(text="What is 2+2?")])
        ])
        msgs = anthropic_to_openai_messages(req)
        assert msgs[0] == {"role": "user", "content": "What is 2+2?"}

    def test_tool_use_block_becomes_tool_calls(self):
        req = _make_request(messages=[
            Message(role="assistant", content=[
                ToolUseBlock(id="tu_1", name="get_weather", input={"city": "Munich"})
            ])
        ])
        msgs = anthropic_to_openai_messages(req)
        assert msgs[0]["role"] == "assistant"
        tc = msgs[0]["tool_calls"][0]
        assert tc["id"] == "tu_1"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Munich"}

    def test_tool_result_block_becomes_tool_message(self):
        req = _make_request(messages=[
            Message(role="user", content=[
                ToolResultBlock(tool_use_id="tu_1", content="15°C, partly cloudy")
            ])
        ])
        msgs = anthropic_to_openai_messages(req)
        assert msgs[0]["role"] == "tool"
        assert msgs[0]["tool_call_id"] == "tu_1"
        assert msgs[0]["content"] == "15°C, partly cloudy"

    def test_tool_result_text_block_list(self):
        req = _make_request(messages=[
            Message(role="user", content=[
                ToolResultBlock(
                    tool_use_id="tu_1",
                    content=[TextBlock(text="Part A"), TextBlock(text="Part B")],
                )
            ])
        ])
        msgs = anthropic_to_openai_messages(req)
        assert msgs[0]["content"] == "Part A Part B"

    def test_mixed_text_and_tool_use(self):
        req = _make_request(messages=[
            Message(role="assistant", content=[
                TextBlock(text="Checking weather..."),
                ToolUseBlock(id="tu_2", name="get_weather", input={}),
            ])
        ])
        msgs = anthropic_to_openai_messages(req)
        assert msgs[0]["content"] == "Checking weather..."
        assert len(msgs[0]["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# anthropic_to_openai_tools
# ---------------------------------------------------------------------------

class TestAnthropicToOpenAITools:
    def test_basic_tool(self):
        req = _make_request(tools=[
            ToolDefinition(
                name="get_weather",
                description="Get current weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ])
        tools = anthropic_to_openai_tools(req)
        assert len(tools) == 1
        fn = tools[0]["function"]
        assert fn["name"] == "get_weather"
        assert fn["parameters"]["properties"]["city"]["type"] == "string"

    def test_empty_tools(self):
        req = _make_request()
        assert anthropic_to_openai_tools(req) == []


# ---------------------------------------------------------------------------
# anthropic_to_openai_tool_choice
# ---------------------------------------------------------------------------

class TestToolChoice:
    def test_auto(self):
        req = _make_request(tool_choice=ToolChoiceAuto())
        assert anthropic_to_openai_tool_choice(req) == "auto"

    def test_specific_tool(self):
        req = _make_request(tool_choice=ToolChoiceTool(name="get_weather"))
        tc = anthropic_to_openai_tool_choice(req)
        assert tc == {"type": "function", "function": {"name": "get_weather"}}

    def test_none(self):
        req = _make_request()
        assert anthropic_to_openai_tool_choice(req) is None


# ---------------------------------------------------------------------------
# openai_to_anthropic_response
# ---------------------------------------------------------------------------

class TestOpenAIToAnthropicResponse:
    def _oai_response(self, content=None, tool_calls=None, finish_reason="stop"):
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {
            "id": "chatcmpl-abc",
            "choices": [{"message": msg, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    def test_text_response(self):
        oai = self._oai_response(content="The answer is 4.")
        resp = openai_to_anthropic_response(oai, "qwen-72b")
        assert resp.stop_reason == "end_turn"
        assert resp.content[0].type == "text"
        assert resp.content[0].text == "The answer is 4."  # type: ignore[union-attr]
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 20

    def test_tool_call_response(self):
        oai = self._oai_response(
            tool_calls=[{
                "id": "tc_1",
                "function": {"name": "get_weather", "arguments": '{"city": "Munich"}'},
            }],
            finish_reason="tool_calls",
        )
        resp = openai_to_anthropic_response(oai, "qwen-72b")
        assert resp.stop_reason == "tool_use"
        tu = resp.content[0]
        assert isinstance(tu, ToolUseBlock)
        assert tu.name == "get_weather"
        assert tu.input == {"city": "Munich"}

    def test_max_tokens_stop_reason(self):
        oai = self._oai_response(content="...", finish_reason="length")
        resp = openai_to_anthropic_response(oai, "qwen-72b")
        assert resp.stop_reason == "max_tokens"

    def test_model_preserved(self):
        oai = self._oai_response(content="hi")
        resp = openai_to_anthropic_response(oai, "my-model")
        assert resp.model == "my-model"
