"""Pydantic schemas for the Anthropic Messages API."""

from __future__ import annotations

from typing import Any, Literal, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list[TextBlock]] = ""


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, list[ContentBlock]]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolDefinition(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"


class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"


class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceTool]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class MessagesRequest(BaseModel):
    model: str
    messages: list[Message]
    system: Union[str, list[TextBlock], None] = None
    max_tokens: int = 4096
    stream: bool = False
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: Union[ToolChoice, None] = None
    temperature: Union[float, None] = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[Union[TextBlock, ToolUseBlock]]
    model: str
    stop_reason: Union[Literal["end_turn", "tool_use", "max_tokens"], None] = None
    stop_sequence: Union[str, None] = None
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------

class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessagesResponse


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Union[TextBlock, ToolUseBlock]


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[TextDelta, InputJsonDelta]


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaPayload(BaseModel):
    stop_reason: Union[Literal["end_turn", "tool_use", "max_tokens"], None] = None
    stop_sequence: Union[str, None] = None


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaPayload
    usage: UsageInfo = Field(default_factory=UsageInfo)


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"
