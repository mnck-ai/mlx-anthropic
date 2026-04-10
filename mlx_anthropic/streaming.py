"""OpenAI SSE chunks → Anthropic SSE events.

Each call to `openai_chunk_to_anthropic_events` is a generator that yields
zero or more Anthropic event dicts for a single OpenAI streaming chunk.

Event lifecycle:
  message_start (once)
  content_block_start (once per content block: text or tool_use)
  content_block_delta* (one or more per block)
  content_block_stop (once per block)
  message_delta (once, with stop_reason)
  message_stop (once)

Caller is responsible for emitting message_start before the first chunk
and flushing the SSE stream after message_stop.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def make_message_start(message_id: str, model: str) -> str:
    return _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })


def openai_chunk_to_anthropic_events(
    chunk: dict[str, Any],
    *,
    text_block_open: bool,
    tool_index: int,
    tool_block_open: bool,
) -> Generator[tuple[str, bool, int, bool], None, None]:
    """Translate one OpenAI streaming chunk to Anthropic events.

    Yields tuples of (sse_string, text_block_open, tool_index, tool_block_open)
    so the caller can track open-block state across chunks.

    Args:
        chunk: Parsed OpenAI chunk dict.
        text_block_open: Whether a text content_block is currently open.
        tool_index: Index of the last opened tool_use block (-1 if none).
        tool_block_open: Whether a tool_use content_block is currently open.
    """
    choices = chunk.get("choices", [])
    if not choices:
        return

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    # --- Text delta ---
    if delta.get("content"):
        if not text_block_open:
            yield (
                _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }),
                True, tool_index, tool_block_open,
            )
            text_block_open = True
        yield (
            _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta["content"]},
            }),
            text_block_open, tool_index, tool_block_open,
        )

    # --- Tool call deltas ---
    for tc_delta in delta.get("tool_calls") or []:
        idx = tc_delta.get("index", 0)
        fn = tc_delta.get("function", {})

        # New tool block starts when we see a name for the first time
        if fn.get("name"):
            # Close previous tool block if any
            if tool_block_open:
                yield (
                    _sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": tool_index + 1,
                    }),
                    text_block_open, tool_index, False,
                )
                tool_block_open = False

            tool_index = idx + 1  # +1 because index 0 is reserved for text
            yield (
                _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": tool_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc_delta.get("id", f"call_{idx}"),
                        "name": fn["name"],
                        "input": {},
                    },
                }),
                text_block_open, tool_index, True,
            )
            tool_block_open = True

        if fn.get("arguments"):
            block_idx = tool_index if tool_block_open else idx + 1
            yield (
                _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                }),
                text_block_open, tool_index, tool_block_open,
            )

    # --- Finish ---
    if finish_reason:
        # Close open blocks
        if text_block_open:
            yield (
                _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
                False, tool_index, tool_block_open,
            )
            text_block_open = False
        if tool_block_open:
            yield (
                _sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": tool_index,
                }),
                text_block_open, tool_index, False,
            )
            tool_block_open = False

        stop_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
        stop_reason = stop_map.get(finish_reason, "end_turn")
        yield (
            _sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }),
            text_block_open, tool_index, tool_block_open,
        )
        yield (
            _sse("message_stop", {"type": "message_stop"}),
            text_block_open, tool_index, tool_block_open,
        )
