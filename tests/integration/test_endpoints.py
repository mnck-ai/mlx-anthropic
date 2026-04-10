"""Integration tests for mlx-anthropic endpoints.

Requires a running server. Set MLX_ANTHROPIC_URL to point at it:
    MLX_ANTHROPIC_URL=http://localhost:8080 pytest tests/integration/

If the env var is not set or the server is unreachable, all tests are skipped.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

BASE_URL = os.environ.get("MLX_ANTHROPIC_URL", "http://localhost:8080").rstrip("/")
TIMEOUT = 120


def _is_server_up() -> bool:
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


skip_if_no_server = pytest.mark.skipif(
    not _is_server_up(),
    reason=f"No server at {BASE_URL} — set MLX_ANTHROPIC_URL and start the server",
)


# ---------------------------------------------------------------------------
# T1: Server health
# ---------------------------------------------------------------------------

@skip_if_no_server
def test_t1_server_health():
    resp = httpx.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# T2: OpenAI /v1/chat/completions — normal non-streaming response
# ---------------------------------------------------------------------------

@skip_if_no_server
def test_t2_openai_chat_completion():
    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "max_tokens": 20,
            "stream": False,
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# T3: Anthropic /v1/messages — non-streaming
# ---------------------------------------------------------------------------

@skip_if_no_server
def test_t3_anthropic_messages_non_streaming():
    resp = httpx.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "max_tokens": 20,
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["stop_reason"] in ("end_turn", "max_tokens")
    assert any(b["type"] == "text" for b in data["content"])


# ---------------------------------------------------------------------------
# T4: Anthropic /v1/messages — non-streaming tool call
# ---------------------------------------------------------------------------

@skip_if_no_server
def test_t4_anthropic_tool_call_non_streaming():
    resp = httpx.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "What is the weather in Munich?"}],
            "max_tokens": 256,
            "tools": [{
                "name": "get_weather",
                "description": "Get current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
            "tool_choice": {"type": "any"},
        },
        timeout=TIMEOUT,
    )
    assert resp.status_code == 200
    data = resp.json()
    tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
    assert tool_blocks, f"Expected tool_use block, got: {data['content']}"
    assert tool_blocks[0]["name"] == "get_weather"
    assert "city" in tool_blocks[0]["input"]


# ---------------------------------------------------------------------------
# T5: Anthropic /v1/messages — streaming text
# ---------------------------------------------------------------------------

@skip_if_no_server
def test_t5_anthropic_streaming_text():
    events = []
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/messages",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "max_tokens": 50,
            "stream": True,
        },
        timeout=TIMEOUT,
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event: "):
                events.append(line.removeprefix("event: "))

    event_types = set(events)
    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types


# ---------------------------------------------------------------------------
# T6: Anthropic /v1/messages — streaming tool call
# ---------------------------------------------------------------------------

@skip_if_no_server
def test_t6_anthropic_streaming_tool_call():
    events: list[tuple[str, dict]] = []
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/messages",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
            "max_tokens": 256,
            "stream": True,
            "tools": [{
                "name": "get_weather",
                "description": "Get current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }],
            "tool_choice": {"type": "any"},
        },
        timeout=TIMEOUT,
    ) as resp:
        assert resp.status_code == 200
        current_event = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ")
            elif line.startswith("data: ") and current_event:
                try:
                    data = json.loads(line.removeprefix("data: "))
                    events.append((current_event, data))
                except json.JSONDecodeError:
                    pass
                current_event = None

    event_types = {e for e, _ in events}
    assert "message_start" in event_types
    assert "message_stop" in event_types

    # At least one content_block_start for a tool_use block
    tool_starts = [
        d for e, d in events
        if e == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use"
    ]
    assert tool_starts, "No tool_use content_block_start found in streaming response"
    assert tool_starts[0]["content_block"]["name"] == "get_weather"

    # message_delta should carry tool_use stop reason
    msg_deltas = [d for e, d in events if e == "message_delta"]
    assert msg_deltas
    assert msg_deltas[-1]["delta"]["stop_reason"] == "tool_use"
