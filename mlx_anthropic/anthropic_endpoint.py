"""POST /v1/messages — Anthropic Messages API handler."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .schema import MessagesRequest, MessagesResponse, TextBlock, ToolUseBlock
from .streaming import make_message_start, openai_chunk_to_anthropic_events
from .tool_parsers.base import ToolParser
from .translation import build_openai_request, openai_to_anthropic_response


async def handle_messages(
    request: MessagesRequest,
    backend_url: str,
    backend_model: str,
    tool_parser: ToolParser | None = None,
) -> MessagesResponse | StreamingResponse:
    """Route a /v1/messages request to a local mlx-lm backend.

    If the backend returns raw text tool calls (not structured tool_calls),
    tool_parser is used to extract and convert them to tool_use blocks.
    """
    oai_body = build_openai_request(request, backend_model)

    if request.stream:
        return StreamingResponse(
            _stream_anthropic(backend_url, oai_body, request.model, tool_parser),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{backend_url}/v1/chat/completions",
            json=oai_body,
            headers={"Authorization": "Bearer local"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    result = openai_to_anthropic_response(resp.json(), request.model)

    # If tools were requested and response has no tool_use blocks,
    # try running the text through the tool parser.
    if request.tools and tool_parser and not any(
        isinstance(b, ToolUseBlock) for b in result.content
    ):
        result = _apply_tool_parser(result, tool_parser)

    return result


def _apply_tool_parser(
    response: MessagesResponse, tool_parser: ToolParser
) -> MessagesResponse:
    """Run text content through the tool parser; replace with tool_use blocks if found."""
    text_content = " ".join(
        b.text for b in response.content if isinstance(b, TextBlock)
    )
    if not text_content:
        return response

    extracted = tool_parser.extract_tool_calls(text_content)
    if not extracted.tools_called:
        return response

    new_content: list[TextBlock | ToolUseBlock] = []
    if extracted.content:
        new_content.append(TextBlock(text=extracted.content))
    for tc in extracted.tool_calls:
        try:
            args = json.loads(tc.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        new_content.append(ToolUseBlock(
            id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            name=tc["name"],
            input=args,
        ))

    return MessagesResponse(
        id=response.id,
        content=new_content,
        model=response.model,
        stop_reason="tool_use",
        usage=response.usage,
    )


async def _stream_anthropic(
    backend_url: str,
    oai_body: dict[str, Any],
    model: str,
    tool_parser: ToolParser | None = None,
) -> AsyncGenerator[str, None]:
    """Stream OAI chunks as Anthropic SSE events.

    When a tool_parser is provided and the backend emits raw text tool calls
    (no structured tool_calls in OAI chunks), we accumulate the full text,
    run it through the parser, and emit synthetic Anthropic tool_use events.
    """
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    yield make_message_start(message_id, model)

    text_open = False
    tool_idx = -1
    tool_open = False
    accumulated_text = ""
    got_tool_calls = False  # True if backend sent structured tool_calls

    oai_body = {**oai_body, "stream": True}
    buffered_events: list[str] = []

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            f"{backend_url}/v1/chat/completions",
            json=oai_body,
            headers={"Authorization": "Bearer local"},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Track whether we see structured tool_calls
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("tool_calls"):
                    got_tool_calls = True
                if delta.get("content"):
                    accumulated_text += delta["content"]

                events_for_chunk = list(openai_chunk_to_anthropic_events(
                    chunk,
                    text_block_open=text_open,
                    tool_index=tool_idx,
                    tool_block_open=tool_open,
                ))

                if not events_for_chunk:
                    continue

                # Update state from last event in chunk
                _, text_open, tool_idx, tool_open = events_for_chunk[-1]

                for sse, *_ in events_for_chunk:
                    if tool_parser and not got_tool_calls:
                        # Buffer events — we may need to rewrite them after parsing
                        buffered_events.append(sse)
                    else:
                        yield sse

    # If we never saw structured tool_calls and have a parser, try parsing accumulated text
    if tool_parser and not got_tool_calls and accumulated_text:
        extracted = tool_parser.extract_tool_calls(accumulated_text)
        if extracted.tools_called:
            # Emit synthetic tool_use events instead of the buffered text events
            for sse in _emit_tool_use_events(extracted.tool_calls):
                yield sse
            return

    # No tool calls parsed — emit buffered events as-is
    for sse in buffered_events:
        yield sse


def _emit_tool_use_events(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Synthesize Anthropic SSE events for parsed tool calls."""
    import json as _json

    events: list[str] = []

    def sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

    for i, tc in enumerate(tool_calls):
        try:
            args = _json.loads(tc.get("arguments", "{}"))
        except _json.JSONDecodeError:
            args = {}
        block_idx = i + 1  # 0 is reserved for text

        events.append(sse("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {
                "type": "tool_use",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": tc["name"],
                "input": {},
            },
        }))
        events.append(sse("content_block_delta", {
            "type": "content_block_delta",
            "index": block_idx,
            "delta": {"type": "input_json_delta", "partial_json": _json.dumps(args)},
        }))
        events.append(sse("content_block_stop", {
            "type": "content_block_stop",
            "index": block_idx,
        }))

    events.append(sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }))
    events.append(sse("message_stop", {"type": "message_stop"}))
    return events
