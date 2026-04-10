"""Unit tests for QwenToolParser."""

import json

import pytest

from mlx_anthropic.tool_parsers.qwen import QwenToolParser


@pytest.fixture
def parser():
    return QwenToolParser()


# ---------------------------------------------------------------------------
# extract_tool_calls
# ---------------------------------------------------------------------------

class TestExtractToolCalls:
    def test_single_brace_xml(self, parser):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Munich"}}\n</tool_call>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_double_brace_xml(self, parser):
        """Qwen2.5 double-brace {{...}} normalization."""
        text = '<tool_call>\n{{"name": "get_weather", "arguments": {"city": "Munich"}}}\n</tool_call>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called, "double-brace patch not applied"
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_triple_brace_xml(self, parser):
        """{{{...}}} triple-brace edge case is also normalized."""
        text = '<tool_call>\n{{{"name": "get_weather", "arguments": {"city": "Munich"}}}}\n</tool_call>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_no_tool_call(self, parser):
        result = parser.extract_tool_calls("The weather in Munich is sunny.")
        assert not result.tools_called
        assert result.tool_calls == []

    def test_arguments_are_json_string(self, parser):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        args = result.tool_calls[0]["arguments"]
        assert isinstance(args, str), "arguments must be a JSON string"
        assert json.loads(args)["city"] == "Berlin"

    def test_bracket_style(self, parser):
        """Qwen3 bracket-style: [Calling tool: func_name({...})]"""
        text = '[Calling tool: get_weather({"city": "Paris"})]'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "get_weather"
        assert json.loads(result.tool_calls[0]["arguments"])["city"] == "Paris"

    def test_think_tags_stripped(self, parser):
        text = '<think>Let me use a tool.</think><tool_call>\n{"name": "search", "arguments": {"q": "mlx"}}\n</tool_call>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["name"] == "search"

    def test_content_returned_when_no_tool(self, parser):
        text = "Here is your answer."
        result = parser.extract_tool_calls(text)
        assert not result.tools_called
        assert result.content == text

    def test_tool_id_generated(self, parser):
        text = '<tool_call>\n{"name": "calc", "arguments": {}}\n</tool_call>'
        result = parser.extract_tool_calls(text)
        assert result.tools_called
        assert result.tool_calls[0]["id"].startswith("call_")


# ---------------------------------------------------------------------------
# extract_tool_calls_streaming
# ---------------------------------------------------------------------------

class TestExtractToolCallsStreaming:
    def test_no_marker_passes_content(self, parser):
        result = parser.extract_tool_calls_streaming(
            previous_text="The weather",
            current_text="The weather in Munich",
            delta_text=" in Munich",
        )
        assert result is not None
        assert result["content"] == " in Munich"

    def test_accumulating_returns_none(self, parser):
        """While inside <tool_call>...</tool_call>, suppress the chunk."""
        result = parser.extract_tool_calls_streaming(
            previous_text='<tool_call>\n{"name": "get_weat',
            current_text='<tool_call>\n{"name": "get_weather", "arguments":',
            delta_text='her", "arguments":',
        )
        assert result is None

    def test_trigger_fires_on_close_tag(self, parser):
        """Trigger fires exactly when </tool_call> first appears in current_text."""
        prev = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Munich"}}'
        curr = prev + "\n</tool_call>"
        result = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=curr,
            delta_text="\n</tool_call>",
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_trigger_no_double_fire(self, parser):
        """Trigger does not fire again when </tool_call> was already in previous_text."""
        prev = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Munich"}}\n</tool_call>'
        result = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev + " ",
            delta_text=" ",
        )
        if result is not None:
            assert "tool_calls" not in result

    def test_streaming_double_brace(self, parser):
        """Streaming trigger with double-brace output parses correctly."""
        prev = '<tool_call>\n{{"name": "get_weather", "arguments": {"city": "Munich"}}}'
        curr = prev + "\n</tool_call>"
        result = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=curr,
            delta_text="\n</tool_call>",
        )
        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
