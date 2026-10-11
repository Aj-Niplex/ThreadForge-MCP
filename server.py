"""
ThreadForge MCP Server
Multi-model conversational chatbot with persistent threads,
model routing/fallback, context compression, and GitHub storage.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STORAGE_REPO = os.getenv("THREADFORGE_STORAGE_REPO", "save-data")
STORAGE_ROOT = os.getenv("THREADFORGE_STORAGE_ROOT", "Chatbot MCP data")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Top-tier model pool for random selection + fallback order
# (Current Gemini models as of Oct 2026)
TOP_TIER_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
]

CHEAP_COMPRESS_MODEL = "gemini-2.5-flash-lite"

mcp = FastMCP(
    name="ThreadForge",
    instructions=(
        "You are ThreadForge, a persistent multi-model chatbot MCP server. "
        "Use the chat tool for conversations. Always continue existing threads "
        "when a thread_id is provided. Save important artifacts deliberately."
    ),
)

# ---------------------------------------------------------------------------
# GitHub storage helpers (via raw GitHub API so we don't depend on Niplex at runtime)
# ---------------------------------------------------------------------------

import httpx

GITHUB_API = "https://api.github.com"
OWNER = "Aj-Niplex"


def _gh_headers() -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _gh_get_file(path: str) -> Optional[dict]:
    """Return {content, sha} or None if not found."""
    url = f"{GITHUB_API}/repos/{OWNER}/{STORAGE_REPO}/contents/{path}"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=_gh_headers())
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        import base64
        content = base64.b64decode(data["content"]).decode("utf-8")
        return {"content": content, "sha": data["sha"]}


def _gh_put_file(path: str, content: str, message: str, sha: Optional[str] = None) -> None:
    url = f"{GITHUB_API}/repos/{OWNER}/{STORAGE_REPO}/contents/{path}"
    import base64
    body: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        body["sha"] = sha
    with httpx.Client(timeout=30) as client:
        r = client.put(url, headers=_gh_headers(), json=body)
        r.raise_for_status()


def _thread_path(thread_id: str, filename: str) -> str:
    return f"{STORAGE_ROOT}/{thread_id}/{filename}"


def load_thread(thread_id: str) -> dict:
    """Load messages + metadata for a thread. Creates empty structure if missing."""
    meta_path = _thread_path(thread_id, "metadata.json")
    msg_path = _thread_path(thread_id, "messages.json")

    meta_raw = _gh_get_file(meta_path)
    msg_raw = _gh_get_file(msg_path)

    if meta_raw is None:
        return {
            "thread_id": thread_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model_preference": None,
            "messages": [],
            "summary": "",
        }

    meta = json.loads(meta_raw["content"])
    messages = json.loads(msg_raw["content"]) if msg_raw else []
    meta["messages"] = messages
    meta["_meta_sha"] = meta_raw["sha"]
    meta["_msg_sha"] = msg_raw["sha"] if msg_raw else None
    return meta


def save_thread(thread: dict) -> None:
    """Persist messages.json, metadata.json and conversation.md."""
    thread_id = thread["thread_id"]
    now = datetime.now(timezone.utc).isoformat()
    thread["updated_at"] = now

    messages = thread.pop("messages", [])
    meta_sha = thread.pop("_meta_sha", None)
    msg_sha = thread.pop("_msg_sha", None)

    # metadata (without messages)
    meta = {k: v for k, v in thread.items() if not k.startswith("_")}
    _gh_put_file(
        _thread_path(thread_id, "metadata.json"),
        json.dumps(meta, indent=2),
        f"ThreadForge: update metadata {thread_id}",
        sha=meta_sha,
    )

    # messages
    _gh_put_file(
        _thread_path(thread_id, "messages.json"),
        json.dumps(messages, indent=2),
        f"ThreadForge: update messages {thread_id}",
        sha=msg_sha,
    )

    # human-readable markdown (best-effort; don't fail the whole save)
    try:
        md_lines = [
            f"# Thread {thread_id}",
            f"Created: {meta.get('created_at', '')}",
            f"Updated: {now}",
            "",
        ]
        for m in messages:
            role = m.get("role", "unknown").upper()
            md_lines.append(f"## {role}")
            md_lines.append(m.get("content", ""))
            md_lines.append("")
        md_path = _thread_path(thread_id, "conversation.md")
        md_existing = _gh_get_file(md_path)
        md_sha = md_existing["sha"] if md_existing else None
        _gh_put_file(
            md_path,
            "\n".join(md_lines),
            f"ThreadForge: update conversation.md {thread_id}",
            sha=md_sha,
        )
    except Exception:
        pass  # messages.json + metadata are the source of truth

    # restore for caller
    thread["messages"] = messages


# ---------------------------------------------------------------------------
# Model calling (Gemini first; easy to extend)
# ---------------------------------------------------------------------------

def call_model(model: str, messages: list[dict], system: str = "") -> str:
    """Call the selected model. Currently Gemini via google-genai."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Convert OpenAI-style messages to Gemini contents
    history = []
    for m in messages[:-1]:  # all but last
        role = "user" if m["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    last = messages[-1]["content"] if messages else ""

    chat = client.chats.create(
        model=model,
        history=history,
        config=types.GenerateContentConfig(
            system_instruction=system or None,
        ) if system else None,
    )
    response = chat.send_message(last)
    return response.text or ""


def choose_model(preferred: Optional[str] = None) -> list[str]:
    """Return ordered list of models to try (preferred first, then random top-tier, then rest)."""
    pool = list(TOP_TIER_MODELS)
    if preferred and preferred in pool:
        pool.remove(preferred)
        return [preferred] + pool
    random.shuffle(pool)
    return pool


def call_with_fallback(messages: list[dict], preferred: Optional[str] = None, system: str = "") -> tuple[str, str]:
    """Try models in order until one succeeds. Returns (reply, model_used)."""
    models = choose_model(preferred)
    last_error = None
    for model in models:
        try:
            reply = call_model(model, messages, system=system)
            if reply and reply.strip():
                return reply, model
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Context compression (cheap model, background style)
# ---------------------------------------------------------------------------

def maybe_compress(thread: dict, max_messages: int = 30) -> None:
    """If history is long, compress older messages into summary using cheap model."""
    messages = thread.get("messages", [])
    if len(messages) <= max_messages:
        return

    # Keep last N, compress the rest
    to_compress = messages[:-12]
    recent = messages[-12:]

    text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
    prompt = [
        {"role": "user", "content": (
            "Summarize the following conversation history into a concise paragraph "
            "that preserves key facts, decisions, and open questions. "
            "Do not add commentary.\n\n" + text
        )}
    ]
    try:
        summary, _ = call_with_fallback(prompt, preferred=CHEAP_COMPRESS_MODEL)
        thread["summary"] = (thread.get("summary") or "") + "\n" + summary
        thread["messages"] = [
            {"role": "system", "content": f"[Compressed earlier context]\n{thread['summary']}"}
        ] + recent
    except Exception:
        # compression is best-effort; keep full history if it fails
        pass


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def chat(
    message: str,
    thread_id: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Send a message to the multi-model chatbot and receive a reply.

    - If thread_id is omitted a new thread is created and returned.
    - If thread_id is provided the full history is loaded and the conversation continues.
    - model: optional preferred model name (otherwise random top-tier + fallback).
    - Returns Markdown containing the reply and the thread_id for continuation.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    thread = load_thread(thread_id)
    preferred = model or thread.get("model_preference")

    # append user message
    thread.setdefault("messages", []).append({
        "role": "user",
        "content": message,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    maybe_compress(thread)

    system = system_prompt or (
        "You are a helpful, concise, and capable AI assistant. "
        "Maintain continuity with the conversation history."
    )
    if thread.get("summary"):
        system += f"\n\nEarlier context summary:\n{thread['summary']}"

    reply, used_model = call_with_fallback(
        thread["messages"],
        preferred=preferred,
        system=system,
    )

    thread["messages"].append({
        "role": "assistant",
        "content": reply,
        "model": used_model,
        "ts": datetime.now(timezone.utc).isoformat(),
    })

    save_thread(thread)

    return (
        f"**Thread ID:** `{thread_id}`\n"
        f"**Model:** `{used_model}`\n\n"
        f"{reply}\n\n"
        f"---\n*Continue this conversation by calling `chat` again with the same thread_id.*"
    )


@mcp.tool()
def list_threads(limit: int = 20) -> str:
    """List recent ThreadForge conversation threads (from the storage repo)."""
    # List the Chatbot MCP data folder via GitHub API
    url = f"{GITHUB_API}/repos/{OWNER}/{STORAGE_REPO}/contents/{STORAGE_ROOT}"
    with httpx.Client(timeout=30) as client:
        r = client.get(url, headers=_gh_headers())
        if r.status_code == 404:
            return "No threads found yet. Start one with the `chat` tool."
        r.raise_for_status()
        items = r.json()

    dirs = [i["name"] for i in items if i["type"] == "dir"]
    dirs = sorted(dirs, reverse=True)[:limit]

    if not dirs:
        return "No threads found yet."

    lines = ["# Recent Threads", ""]
    for tid in dirs:
        lines.append(f"- `{tid}`")
    return "\n".join(lines)


@mcp.tool()
def get_thread(thread_id: str, full: bool = False) -> str:
    """
    Retrieve a thread's summary or full conversation history.

    full=False → metadata + summary + last few messages
    full=True  → entire conversation as Markdown
    """
    thread = load_thread(thread_id)
    if not thread.get("messages") and not thread.get("created_at"):
        return f"Thread `{thread_id}` not found."

    if full:
        md = [f"# Thread {thread_id}", ""]
        for m in thread.get("messages", []):
            md.append(f"## {m.get('role', '?').upper()}")
            md.append(m.get("content", ""))
            md.append("")
        return "\n".join(md)

    lines = [
        f"# Thread `{thread_id}`",
        f"- Created: {thread.get('created_at', '?')}",
        f"- Updated: {thread.get('updated_at', '?')}",
        f"- Preference: {thread.get('model_preference') or 'random top-tier'}",
        f"- Messages: {len(thread.get('messages', []))}",
        "",
        "## Summary",
        thread.get("summary") or "_(none yet)_",
        "",
        "## Last messages",
    ]
    for m in thread.get("messages", [])[-6:]:
        lines.append(f"**{m.get('role')}**: {m.get('content', '')[:300]}...")
    return "\n".join(lines)


@mcp.tool()
def save_artifact(
    thread_id: str,
    title: str,
    content: str,
    kind: str = "note",
) -> str:
    """
    Deliberately save an artifact (note, code, decision, reference, etc.)
    inside the thread's artifacts/ folder.
    """
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:80]
    filename = f"{kind}_{safe_title.replace(' ', '_')}.md"
    path = _thread_path(thread_id, f"artifacts/{filename}")

    body = f"# {title}\n\nKind: {kind}\nThread: {thread_id}\n\n{content}\n"
    _gh_put_file(path, body, f"ThreadForge: save artifact {title} in {thread_id}")
    return f"Artifact saved: `{path}`"


@mcp.tool()
def set_model_preference(thread_id: str, model: Optional[str] = None) -> str:
    """
    Set (or clear) the preferred model for a specific thread.
    Pass model=None to revert to random top-tier selection.
    """
    thread = load_thread(thread_id)
    thread["model_preference"] = model
    save_thread(thread)
    if model:
        return f"Thread `{thread_id}` will prefer model `{model}` (with fallback)."
    return f"Thread `{thread_id}` will use random top-tier models."


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Horizon / production usually expects streamable-http; local dev can use stdio
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)
