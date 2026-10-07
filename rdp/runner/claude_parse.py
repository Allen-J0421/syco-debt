"""Read Claude Code ``stream-json`` output: final reply text and resource use."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator


def _non_empty_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def iter_jsonl_objects(jsonl_text: str) -> Iterator[dict]:
    for raw in jsonl_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _assistant_text(obj: dict) -> str | None:
    if obj.get("type") != "assistant":
        return None
    message = obj.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "output_text"):
            text = _non_empty_str(block.get("text"))
            if text:
                parts.append(text)
        elif block.get("type") == "thinking":
            text = _non_empty_str(block.get("thinking")) or _non_empty_str(block.get("text"))
            if text:
                parts.append(text)
    return "\n".join(parts) if parts else None


def _result_text(obj: dict) -> str | None:
    if obj.get("type") != "result":
        return None
    return _non_empty_str(obj.get("result"))


def extract_final_agent_message(jsonl_text: str) -> str:
    """Last agent reply in file order (assistant block or result line)."""
    last: str | None = None
    for obj in iter_jsonl_objects(jsonl_text):
        text = _assistant_text(obj) or _result_text(obj)
        if text:
            last = text
    return last or ""


_GIT_HISTORY_RE = re.compile(
    r"\bgit\b[^|;&\n]*?\s(log|show|reflog|fetch|pull|clone|remote|blame|ls-remote|rev-list|cat-file|for-each-ref|branch)\b"
)
_WEB_CMD_RE = re.compile(r"\b(curl|wget|gh)\s")
_WEB_TOOLS = {"WebFetch", "WebSearch"}


def probe_counts(jsonl_text: str) -> dict:
    """How often the agent looked outside the code it was given during one turn.

    git_history_calls: shell commands that read git history, refs or remotes.
    web_calls: web tools, or shell commands that fetch from the network.
    The workspace has no history to find, so these are an audit trail, not a control."""
    git_calls = web_calls = 0
    for obj in iter_jsonl_objects(jsonl_text):
        if obj.get("type") != "assistant":
            continue
        content = (obj.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") in _WEB_TOOLS:
                web_calls += 1
                continue
            command = (block.get("input") or {}).get("command")
            if isinstance(command, str):
                git_calls += len(_GIT_HISTORY_RE.findall(command))
                web_calls += len(_WEB_CMD_RE.findall(command))
    return {"git_history_calls": git_calls, "web_calls": web_calls}


_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def extract_usage(jsonl_text: str) -> dict:
    """Sum resource use over every ``result`` line of a turn (two when a clarification follow-up ran)."""
    out: dict = {"agent_calls": 0, "cost_usd": 0.0, "api_duration_ms": 0, "agent_turns": 0}
    out.update({k: 0 for k in _USAGE_KEYS})
    by_model: dict[str, dict] = {}     # the CLI also reports use per model (helper models included)
    seen = False
    for obj in iter_jsonl_objects(jsonl_text):
        if obj.get("type") != "result":
            continue
        seen = True
        out["agent_calls"] += 1
        cost = obj.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            out["cost_usd"] += float(cost)
        for src, dst in (("duration_api_ms", "api_duration_ms"), ("num_turns", "agent_turns")):
            value = obj.get(src)
            if isinstance(value, (int, float)):
                out[dst] += int(value)
        usage = obj.get("usage")
        if isinstance(usage, dict):
            for key in _USAGE_KEYS:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    out[key] += int(value)
        model_usage = obj.get("modelUsage")
        if isinstance(model_usage, dict):
            for model, fields in model_usage.items():
                if not isinstance(fields, dict):
                    continue
                slot = by_model.setdefault(str(model), {})
                for key, value in fields.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        slot[key] = slot.get(key, 0) + value
    out["total_tokens"] = sum(out[k] for k in _USAGE_KEYS)
    out["by_model"] = by_model
    return out if seen else {}
