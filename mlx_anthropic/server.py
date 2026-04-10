"""mlx-anthropic server — Anthropic + OpenAI API endpoints backed by mlx-lm.

Entry point:
    python -m mlx_anthropic.server --model <path-or-repo> [--port 8080]

Flags:
    --model            HuggingFace repo ID or local path (required)
    --host             Bind address (default: 0.0.0.0)
    --port             Port (default: 8080)
    --tool-call-parser Parser name: qwen | qwen3 | auto (default: auto)
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .anthropic_endpoint import handle_messages
from .schema import MessagesRequest
from .tool_parsers.registry import auto_detect

# ---------------------------------------------------------------------------
# Model routing — maps logical names to local backend URLs.
# In standalone mode (mlx-lm subprocess), only one backend is running.
# In multi-model mode, the caller can pass --backend-url explicitly.
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing to do for now; mlx-lm model loading happens in the subprocess
    yield


app = FastAPI(title="mlx-anthropic", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Anthropic endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    try:
        req = MessagesRequest(**body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    backend_url = _state.get("backend_url", "http://localhost:8081")
    backend_model = _state.get("backend_model", req.model)

    tool_parser = _state.get("tool_parser")
    result = await handle_messages(req, backend_url, backend_model, tool_parser=tool_parser)
    if isinstance(result, StreamingResponse):
        return result
    return JSONResponse(result.model_dump())


# ---------------------------------------------------------------------------
# OpenAI pass-through (for compatibility)
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    backend_url = _state.get("backend_url", "http://localhost:8081")
    # Substitute backend model name so mlx-lm doesn't try to load an unknown model
    body = {**body, "model": _state.get("backend_model", body.get("model", "local"))}
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{backend_url}/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer local"},
        )
    return JSONResponse(resp.json(), status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def models():
    model_id = _state.get("model_id", "local-model")
    return JSONResponse({
        "object": "list",
        "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "local"}],
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="mlx-anthropic server")
    parser.add_argument("--model", required=True, help="HuggingFace repo or local path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend-url", default=None,
                        help="URL of running mlx-lm server (default: auto-start on :8081)")
    parser.add_argument("--tool-call-parser", default="auto",
                        choices=["auto", "qwen", "qwen3", "none"])
    args = parser.parse_args()

    _state["model_id"] = args.model
    _state["backend_url"] = args.backend_url or "http://localhost:8081"
    _state["backend_model"] = args.model

    # Resolve tool parser
    if args.tool_call_parser == "auto":
        parser_cls = auto_detect(args.model)
    elif args.tool_call_parser == "none":
        parser_cls = None
    else:
        from .tool_parsers.registry import get_parser  # noqa: PLC0415
        parser_cls = get_parser(args.tool_call_parser)

    if parser_cls:
        _state["tool_parser"] = parser_cls()
        print(f"Tool parser: {parser_cls.__name__}", file=sys.stderr)
    else:
        print("Tool parser: none", file=sys.stderr)

    print(f"Backend: {_state['backend_url']}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
