"""Tool parser registry — maps model name patterns to parser classes."""

from __future__ import annotations

from .base import ToolParser

_REGISTRY: dict[str, type[ToolParser]] = {}


def register(*names: str):
    """Decorator that registers a parser under one or more names."""
    def decorator(cls: type[ToolParser]) -> type[ToolParser]:
        for name in names:
            _REGISTRY[name] = cls
        return cls
    return decorator


def get_parser(name: str) -> type[ToolParser]:
    if name not in _REGISTRY:
        raise KeyError(f"No tool parser registered for '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def auto_detect(model_name: str) -> type[ToolParser] | None:
    """Return a parser class based on model name heuristics, or None."""
    lower = model_name.lower()
    if "qwen" in lower:
        from .qwen import QwenToolParser  # noqa: PLC0415
        return QwenToolParser
    return None
