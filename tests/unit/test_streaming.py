"""Unit tests for mlx_anthropic.streaming."""

import json
import pytest

from mlx_anthropic.streaming import make_message_start, openai_chunk_to_anthropic_events


def _run_chunks(chunks: list[dict]) -> list[tuple[str, dict]]:
    """Run a sequence of OAI chunks through the translator, return (event, data) pairs."""
    text_open = False
    tool_idx = -1
    tool_open = False
    events = []
    for chunk in chunks:
        for sse, text_open, tool_idx, tool_open in openai_chunk_to_anthropic_events(
            chunk,
            text_block_open=text_open,
            tool_index=tool_idx,
            tool_block_open=tool_open,
        ):
            # Parse the SSE string back into (event, data)
            lines = sse.strip().split("\n")
            event = lines[0].removeprefix("event: ")
            data = json.loads(lines[1].removeprefix("data: "))
            events.append((event, data))
    return events


def _make_text_chunk(content: str, finish: str | None = None) -> dict:
    return {
        "choices": [{
            "delta": {"content": content},
            "finish_reason": finish,
        }]
    }


def _make_tool_chunk(index: int, id: str, name: str, args: str = "", finish: str | None = None) -> dict:
    return {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": index,
                    "id": id,
                    "function": {"name": name, "arguments": args},
                }]
            },
            "finish_reason": finish,
        }]
    }


def _make_args_chunk(index: int, args: str, finish: str | None = None) -> dict:
    return {
        "choices": [{
            "delta": {
                "tool_calls": [{"index": index, "function": {"arguments": args}}]
            },
            "finish_reason": finish,
        }]
    }


def _make_finish_chunk(finish: str) -> dict:
    return {"choices": [{"delta": {}, "finish_reason": finish}]}


class TestMakeMessageStart:
    def test_structure(self):
        sse = make_message_start("msg_abc", "qwen-72b")
        lines = sse.strip().split("\n")
        assert lines[0] == "event: message_start"
        data = json.loads(lines[1].removeprefix("data: "))
        assert data["type"] == "message_start"
        assert data["message"]["id"] == "msg_abc"
        assert data["message"]["model"] == "qwen-72b"


class TestTextOnlyStream:
    def test_event_sequence(self):
        chunks = [
            _make_text_chunk("Hello"),
            _make_text_chunk(" world"),
            _make_text_chunk("", finish="stop"),
        ]
        events = _run_chunks(chunks)
        types = [e for e, _ in events]
        assert types == [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]

    def test_text_deltas(self):
        chunks = [_make_text_chunk("Hi"), _make_text_chunk("", finish="stop")]
        events = _run_chunks(chunks)
        delta_events = [(e, d) for e, d in events if e == "content_block_delta"]
        assert delta_events[0][1]["delta"]["text"] == "Hi"

    def test_stop_reason_end_turn(self):
        chunks = [_make_text_chunk("x"), _make_finish_chunk("stop")]
        events = _run_chunks(chunks)
        msg_delta = next(d for e, d in events if e == "message_delta")
        assert msg_delta["delta"]["stop_reason"] == "end_turn"

    def test_stop_reason_max_tokens(self):
        chunks = [_make_text_chunk("x"), _make_finish_chunk("length")]
        events = _run_chunks(chunks)
        msg_delta = next(d for e, d in events if e == "message_delta")
        assert msg_delta["delta"]["stop_reason"] == "max_tokens"


class TestToolCallStream:
    def test_event_sequence(self):
        chunks = [
            _make_tool_chunk(0, "call_1", "get_weather", '{"city":'),
            _make_args_chunk(0, '"Munich"}'),
            _make_finish_chunk("tool_calls"),
        ]
        events = _run_chunks(chunks)
        types = [e for e, _ in events]
        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        assert "message_stop" in types

    def test_tool_block_start_has_name(self):
        chunks = [
            _make_tool_chunk(0, "call_1", "get_weather"),
            _make_finish_chunk("tool_calls"),
        ]
        events = _run_chunks(chunks)
        start = next(d for e, d in events if e == "content_block_start")
        assert start["content_block"]["name"] == "get_weather"
        assert start["content_block"]["type"] == "tool_use"

    def test_args_delta_type(self):
        chunks = [
            _make_tool_chunk(0, "call_1", "fn", '{"a":'),
            _make_args_chunk(0, '1}'),
            _make_finish_chunk("tool_calls"),
        ]
        events = _run_chunks(chunks)
        deltas = [d for e, d in events if e == "content_block_delta"]
        assert all(d["delta"]["type"] == "input_json_delta" for d in deltas)

    def test_stop_reason_tool_use(self):
        chunks = [
            _make_tool_chunk(0, "call_1", "fn"),
            _make_finish_chunk("tool_calls"),
        ]
        events = _run_chunks(chunks)
        msg_delta = next(d for e, d in events if e == "message_delta")
        assert msg_delta["delta"]["stop_reason"] == "tool_use"

    def test_empty_chunk_produces_no_events(self):
        events = _run_chunks([{"choices": []}])
        assert events == []
