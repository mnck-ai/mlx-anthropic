"""Abstract base class for tool call parsers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

_THINK_FULL = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_IMPLICIT = re.compile(r"^.*?</think>", re.DOTALL)


@dataclass
class ExtractedToolCallInformation:
    tools_called: bool
    tool_calls: list[dict[str, Any]]
    content: str | None = None


class ToolParser(ABC):
    def __init__(self) -> None:
        self.current_tool_id: int = -1
        self.prev_tool_call_arr: list[dict[str, Any]] = []

    @staticmethod
    def strip_think_tags(text: str) -> str:
        result = _THINK_FULL.sub("", text)
        if result == text and "</think>" in text:
            result = _THINK_IMPLICIT.sub("", text)
        return result.strip()

    @abstractmethod
    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation: ...

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return None

    def reset(self) -> None:
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
