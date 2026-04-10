# mlx-anthropic

Anthropic Messages API endpoint for [mlx-lm](https://github.com/ml-explore/mlx-lm) — run Claude-compatible models locally on Apple Silicon.

## What it does

`mlx-anthropic` sits in front of a running `mlx-lm` server and translates between the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) and the OpenAI-compatible API that `mlx-lm` exposes. Any client that speaks the Anthropic API — including Claude Code — can point at it with no changes.

```
Client (Anthropic API)
        │
        ▼
 mlx-anthropic :8080
  ├── POST /v1/messages        ← Anthropic Messages API
  ├── POST /v1/chat/completions ← OpenAI pass-through
  └── GET  /v1/models
        │
        ▼
  mlx-lm server :8081
  (Qwen2.5, Gemma, DeepSeek, …)
```

### Features

- Full Anthropic Messages API: text, tool use, streaming
- OpenAI pass-through for hybrid clients
- **Tool call parsing** — handles models whose backends return raw text tool calls (e.g. Qwen XML format `<tool_call>…</tool_call>`) and converts them to structured `tool_use` blocks
- Streaming: OpenAI SSE chunks → Anthropic SSE event sequence
- Auto-detects tool parser from model name (`qwen` → `QwenToolParser`)
- No inference code — `mlx-lm` is a pip dependency, never vendored

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.10+
- A running `mlx-lm` server (see [mlx-lm docs](https://github.com/ml-explore/mlx-lm))

## Installation

```bash
pip install mlx-anthropic
```

Or from source:

```bash
git clone https://github.com/mnck-ai/mlx-anthropic
cd mlx-anthropic
pip install -e .
```

## Usage

### 1. Start an mlx-lm server

```bash
mlx_lm.server --model mlx-community/Qwen2.5-72B-Instruct-4bit --port 8081
```

### 2. Start mlx-anthropic

```bash
python -m mlx_anthropic.server \
  --model mlx-community/Qwen2.5-72B-Instruct-4bit \
  --backend-url http://localhost:8081 \
  --port 8080
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | HuggingFace repo ID or local path |
| `--backend-url` | `http://localhost:8081` | URL of the running mlx-lm server |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Port |
| `--tool-call-parser` | `auto` | `auto` \| `qwen` \| `qwen3` \| `none` |

### 3. Point your client at it

**Claude Code:**
```bash
ANTHROPIC_BASE_URL=http://localhost:8080 claude --model local "What is 2+2?"
```

**Python SDK:**
```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="local")
response = client.messages.create(
    model="local",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
```

**Tool use:**
```python
response = client.messages.create(
    model="local",
    max_tokens=256,
    tools=[{
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }],
    messages=[{"role": "user", "content": "What is the weather in Munich?"}],
)
```

## Tool call parsing

`mlx-lm` does not parse tool calls — it returns raw model output. `mlx-anthropic` adds a parser layer that detects and converts model-specific tool call formats into structured Anthropic `tool_use` blocks.

Currently supported:

| Parser | Models | Format |
|--------|--------|--------|
| `qwen` / `qwen3` | Qwen2.5, Qwen3 | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` |
| | Qwen3 | `[Calling tool: func({…})]` |

The `qwen` parser also handles a Qwen2.5 quirk where the chat template emits `{{…}}` (double-braced JSON) instead of `{…}`.

Pass `--tool-call-parser none` to disable parsing (useful when the backend already returns structured `tool_calls` in OpenAI format).

## Architecture

```
mlx_anthropic/
├── server.py              # FastAPI app + CLI entry point
├── anthropic_endpoint.py  # POST /v1/messages handler + SSE translation
├── schema.py              # Pydantic: MessagesRequest/Response, content blocks, events
├── translation.py         # Anthropic ↔ OpenAI message/tool schema conversion
├── streaming.py           # OpenAI SSE chunks → Anthropic SSE events
└── tool_parsers/
    ├── base.py            # ToolParser ABC + ExtractedToolCallInformation
    ├── registry.py        # Name → parser class registry + auto-detect
    └── qwen.py            # QwenToolParser
```

## Development

```bash
pip install -e ".[dev]"

# Unit tests (no server required)
pytest tests/unit/

# Integration tests (requires running server)
MLX_ANTHROPIC_URL=http://localhost:8080 pytest tests/integration/
```

## Related

- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — inference backend (upstream)
- Separate PR to ml-explore/mlx-lm: `QwenToolParser` fix for issues [#905](https://github.com/ml-explore/mlx-lm/issues/905) and [#1096](https://github.com/ml-explore/mlx-lm/issues/1096)

## License

MIT
