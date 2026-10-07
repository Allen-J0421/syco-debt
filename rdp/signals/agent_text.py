"""Load agent thinking + message text from per-turn run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from rdp.signals.keyword_regex import agent_text


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def claude_extract(rundir: Path) -> tuple[list[str], list[str]]:
    """Return (thinking_blocks, message_texts) for a claude run."""
    thinking: list[str] = []
    texts: list[str] = []
    for o in load_jsonl(rundir / "claude.jsonl"):
        if not isinstance(o, dict) or o.get("type") != "assistant":
            continue
        for b in o.get("message", {}).get("content", []):
            bt = b.get("type")
            if bt == "thinking" and b.get("thinking"):
                thinking.append(b["thinking"].strip())
            elif bt == "text" and b.get("text"):
                texts.append(b["text"].strip())
    return thinking, texts


def codex_extract(rundir: Path) -> tuple[list[str], list[str]]:
    """Return (thinking_blocks, message_texts) for a codex/gpt run."""
    texts: list[str] = []
    for o in load_jsonl(rundir / "codex.jsonl"):
        if not isinstance(o, dict):
            continue
        if o.get("type") == "item.completed":
            it = o.get("item", {})
            if it.get("type") == "agent_message" and it.get("text"):
                texts.append(it["text"].strip())
    return [], texts


def extract_agent_text(rundir: Path, model: str) -> str:
    """Concatenate thinking + messages for one turn's run directory."""
    if model.strip().lower().startswith("claude"):
        thinking, messages = claude_extract(rundir)
    else:
        thinking, messages = codex_extract(rundir)
    return agent_text(thinking, messages)
