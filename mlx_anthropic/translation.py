"""Anthropic ↔ OpenAI message and schema translation."""

from __future__ import annotations

import json
from typing import Any, Literal

from .schema import (
    MessagesRequest, MessagesResponse, UsageInfo,
    TextBlock, ToolUseBlock, ToolResultBlock,
)


def anthropic_to_openai_messages(request: MessagesRequest) -> list[dict[str, Any]]:
    """Convert Anthropic messages (+ system prompt) to OpenAI chat messages."""
    messages: list[dict[str, Any]] = []

    # System prompt
    if request.system:
        if isinstance(request.system, str):
            messages.append({"role": "system", "content": request.system})
        else:
            # list[TextBlock]
            text = " ".join(b.text for b in request.system)
            messages.append({"role": "system", "content": text})

    for m in request.messages:
        content = m.content

        if isinstance(content, str):
            messages.append({"role": m.role, "content": content})
            continue

        # content is list[ContentBlock]
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })
            elif isinstance(block, ToolResultBlock):
                result_content = block.content
                if isinstance(result_content, list):
                    result_content = " ".join(b.text for b in result_content)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.tool_use_id,
                    "content": result_content,
                })

        if tool_calls:
            # Assistant message with tool calls — content may be empty
            msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg["content"] = " ".join(text_parts)
            else:
                msg["content"] = None
            msg["tool_calls"] = tool_calls
            messages.append(msg)
        elif text_parts:
            messages.append({"role": m.role, "content": " ".join(text_parts)})

        messages.extend(tool_results)

    return messages


def anthropic_to_openai_tools(request: MessagesRequest) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function tools."""
    tools = []
    for t in request.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        })
    return tools


def anthropic_to_openai_tool_choice(request: MessagesRequest) -> str | dict[str, Any] | None:
    """Convert Anthropic tool_choice to OpenAI tool_choice."""
    if not request.tool_choice:
        return None
    tc = request.tool_choice
    if tc.type == "auto":
        return "auto"
    if tc.type == "any":
        return "required"
    if tc.type == "tool":
        return {"type": "function", "function": {"name": tc.name}}
    return None


def build_openai_request(request: MessagesRequest, model: str) -> dict[str, Any]:
    """Build a complete OpenAI /v1/chat/completions request body."""
    body: dict[str, Any] = {
        "model": model,
        "messages": anthropic_to_openai_messages(request),
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.tools:
        body["tools"] = anthropic_to_openai_tools(request)
        tc = anthropic_to_openai_tool_choice(request)
        if tc is not None:
            body["tool_choice"] = tc
    return body


def openai_to_anthropic_response(oai: dict[str, Any], model: str) -> MessagesResponse:
    """Convert an OpenAI ChatCompletion response to an Anthropic MessagesResponse."""
    choice = oai["choices"][0]
    msg = choice["message"]
    content: list[TextBlock | ToolUseBlock] = []

    if msg.get("content"):
        content.append(TextBlock(text=msg["content"]))

    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except Exception:
            args = {}
        content.append(ToolUseBlock(id=tc["id"], name=fn["name"], input=args))

    _finish = choice.get("finish_reason", "stop")
    stop_reason: Literal["end_turn", "tool_use", "max_tokens"]
    if _finish == "tool_calls":
        stop_reason = "tool_use"
    elif _finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage_raw = oai.get("usage", {})
    return MessagesResponse(
        id=oai.get("id", "msg_local"),
        content=content,
        model=model,
        stop_reason=stop_reason,
        usage=UsageInfo(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
        ),
    )
