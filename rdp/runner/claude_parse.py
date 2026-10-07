"""Read Claude Code ``stream-json`` output: final reply text and resource use."""

from __future__ import annotations

import json
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


_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def extract_usage(jsonl_text: str) -> dict:
    """Sum resource use over every ``result`` line of a turn (two when a clarification follow-up ran)."""
    out: dict = {"agent_calls": 0, "cost_usd": 0.0, "api_duration_ms": 0, "agent_turns": 0}
    out.update({k: 0 for k in _USAGE_KEYS})
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
    return out if seen else {}
