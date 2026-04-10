"""POST /v1/messages — Anthropic Messages API handler."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .schema import MessagesRequest, MessagesResponse
from .streaming import make_message_start, openai_chunk_to_anthropic_events
from .translation import build_openai_request, openai_to_anthropic_response


async def handle_messages(
    request: MessagesRequest,
    backend_url: str,
    backend_model: str,
) -> MessagesResponse | StreamingResponse:
    """Route a /v1/messages request to a local mlx-lm backend."""
    oai_body = build_openai_request(request, backend_model)

    if request.stream:
        return StreamingResponse(
            _stream_anthropic(backend_url, oai_body, request.model),
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

    return openai_to_anthropic_response(resp.json(), request.model)


async def _stream_anthropic(
    backend_url: str,
    oai_body: dict,
    model: str,
) -> AsyncGenerator[str, None]:
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    yield make_message_start(message_id, model)

    text_open = False
    tool_idx = -1
    tool_open = False

    oai_body = {**oai_body, "stream": True}
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
                for sse, text_open, tool_idx, tool_open in openai_chunk_to_anthropic_events(
                    chunk,
                    text_block_open=text_open,
                    tool_index=tool_idx,
                    tool_block_open=tool_open,
                ):
                    yield sse
