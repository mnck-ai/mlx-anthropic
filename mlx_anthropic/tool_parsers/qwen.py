"""Tool call parser for Qwen models.

Supports two Qwen tool calling formats:
- XML:     <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Bracket: [Calling tool: func_name({"arg": "value"})]

Key fix: Qwen2.5's chat template emits {{...}} (double outer braces) in tool
call XML due to Jinja2 template escaping. We try raw JSON first; on failure,
strip one brace layer if the string starts with {{ and ends with }}.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from .base import ExtractedToolCallInformation, ToolParser
from .registry import register


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


@register("qwen", "qwen3")
class QwenToolParser(ToolParser):
    XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*\})\s*</tool_call>", re.DOTALL)
    BRACKET_PATTERN = re.compile(r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        tool_calls: list[dict[str, Any]] = []
        cleaned = self.strip_think_tags(model_output)

        # Bracket style (Qwen3)
        for name, args_str in self.BRACKET_PATTERN.findall(cleaned):
            try:
                arguments = json.loads(args_str)
                tool_calls.append({
                    "id": _generate_tool_id(),
                    "name": name.strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False)
                    if isinstance(arguments, dict) else str(arguments),
                })
            except json.JSONDecodeError:
                continue
        if tool_calls:
            cleaned = self.BRACKET_PATTERN.sub("", cleaned).strip()

        # XML style (Qwen2.5 + Qwen3 fallback)
        for match in self.XML_PATTERN.findall(cleaned):
            data = self._parse_possibly_double_braced(match)
            if data is None:
                continue
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if not name:
                continue
            tool_calls.append({
                "id": _generate_tool_id(),
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False)
                if isinstance(arguments, dict) else str(arguments),
            })
        if self.XML_PATTERN.search(cleaned):
            cleaned = self.XML_PATTERN.sub("", cleaned).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned or None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    @staticmethod
    def _parse_possibly_double_braced(s: str) -> dict[str, Any] | None:
        """Try json.loads; on failure strip outer brace layers one at a time."""
        candidate = s
        while True:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                if candidate.startswith("{{") and candidate.endswith("}}"):
                    candidate = candidate[1:-1]
                else:
                    return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        has_marker = "<tool_call>" in current_text or "[Calling tool:" in current_text
        if not has_marker:
            return {"content": delta_text}

        complete = (
            ("</tool_call>" in current_text and "</tool_call>" not in previous_text) or
            (")]" in current_text and ")]" not in previous_text)
        )
        if complete:
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }
        return None
