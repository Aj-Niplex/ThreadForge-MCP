# ThreadForge MCP

**Multi-model conversational chatbot MCP server** with persistent threads, model routing + fallback, context compression, and GitHub-backed storage.

Connectable by Claude, Grok, Mistral, Gemini hosts, and any MCP client. Designed for **Horizon Prefect** deployment and later integration into **NiPlex MCP** (so Claude can reach it via the single-connector limit).

## Features

- True multi-turn chat (not one-shot Q&A)
- Thread IDs with full history stored in the `save-data` GitHub repo
- Random / explicit top-tier model selection + automatic fallback on errors
- Background context compression with cheap models so windows never overflow
- Markdown responses + full conversation saved as Markdown
- Works standalone or as a downstream server under NiPlex MCP

## Quick Start (local)

```bash
uv sync
export GEMINI_API_KEY=your_key_here
# optional: GITHUB_TOKEN if not using Horizon secrets
uv run python server.py
```

Or with the MCP Inspector:

```bash
uv run mcp dev server.py
```

## Horizon Prefect

1. Connect this repo in [horizon.prefect.io](https://horizon.prefect.io)
2. Set env vars: `GEMINI_API_KEY`, and any other model keys
3. Deploy → get a live MCP URL with OAuth

## Tools

| Tool | Purpose |
|------|---------|
| `chat` | Main conversation entrypoint (message + optional thread_id + optional model) |
| `list_threads` | List recent threads |
| `get_thread` | Fetch history / summary of a thread |
| `save_artifact` | Let the AI deliberately store notes, code, decisions, etc. |
| `set_model_preference` | Per-thread or global model preference |

## Storage layout (in `save-data` repo)

```
Chatbot MCP data/
├─ {thread_id}/
│   ├─ messages.json
│   ├─ conversation.md
│   ├─ summary.md
│   ├─ metadata.json
│   └─ artifacts/
```

## NiPlex integration (later)

Once stable, expose ThreadForge tools through NiPlex MCP so Claude Desktop / Claude.ai can use the chatbot via the single connector.

---
Built for Adarsh / NiPlex.
