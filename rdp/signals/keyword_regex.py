"""Shared keyword-regex utilities for verbal-refusal detection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FLAG_MAP = {"IGNORECASE": re.I, "MULTILINE": re.M, "DOTALL": re.S, "VERBOSE": re.X}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "keyword_regex_config.json"


def load_config(path: str | Path) -> tuple[dict[str, Any], dict, list[str]]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    flags = 0
    for flag_name in cfg.get("options", {}).get("flags", ["IGNORECASE"]):
        flags |= FLAG_MAP[flag_name]
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
    for cat, pats in cfg["categories"].items():
        compiled[cat] = [(p, re.compile(p, flags)) for p in pats]
    decline_cats = cfg["options"].get("decline_categories", list(cfg["categories"]))
    return cfg, compiled, decline_cats


def agent_text(thinking: list[str], messages: list[str]) -> str:
    return "\n".join(thinking + messages)


def agent_text_from_record(record: dict) -> str:
    return agent_text(record.get("thinking", []), record.get("messages", []))


def matched_categories(
    text: str, compiled: dict[str, list[tuple[str, re.Pattern[str]]]]
) -> dict[str, list[str]]:
    """Return {category: [matching pattern strings]} for a piece of text."""
    hits: dict[str, list[str]] = {}
    for cat, pats in compiled.items():
        matched = [p for p, rx in pats if rx.search(text)]
        if matched:
            hits[cat] = matched
    return hits


def is_verbal_decline(
    text: str,
    compiled: dict[str, list[tuple[str, re.Pattern[str]]]],
    decline_cats: list[str],
) -> bool:
    hits = matched_categories(text, compiled)
    return any(cat in hits for cat in decline_cats)
