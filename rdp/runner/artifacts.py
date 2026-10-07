"""Per-run artifact writers."""

from __future__ import annotations

import re
from pathlib import Path

PROMPT_FILE_NAME = "prompt.txt"
PROMPT_TURN_HEADER_RE = re.compile(r"^=== Turn (\d+) ===\s*$", re.MULTILINE)


def format_prompt_turn(turn: int, text: str) -> str:
    return f"=== Turn {turn} ===\n{text.strip()}"


def format_clarification_reply(turn: int, text: str) -> str:
    return f"=== Turn {turn} (clarification reply) ===\n{text.strip()}"


def append_clarification_reply(prompt_path: Path, turn: int, text: str) -> None:
    stripped = text.strip()
    if not stripped:
        return
    block = format_clarification_reply(turn, stripped)
    if prompt_path.exists() and prompt_path.stat().st_size > 0:
        content = prompt_path.read_text(encoding="utf-8").rstrip() + "\n\n" + block + "\n"
    else:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        content = block + "\n"
    prompt_path.write_text(content, encoding="utf-8")


def append_prompt_turn(prompt_path: Path, turn: int, text: str) -> None:
    """Append one labeled Gemini prompt to the experiment-level prompt file."""
    stripped = text.strip()
    if not stripped:
        return
    block = format_prompt_turn(turn, stripped)
    if prompt_path.exists() and prompt_path.stat().st_size > 0:
        content = prompt_path.read_text(encoding="utf-8").rstrip() + "\n\n" + block + "\n"
    else:
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        content = block + "\n"
    prompt_path.write_text(content, encoding="utf-8")


def parse_prompts_file(text: str) -> dict[int, str]:
    """Parse experiment-level prompt.txt into turn -> prompt text."""
    matches = list(PROMPT_TURN_HEADER_RE.finditer(text))
    if not matches:
        return {}

    prompts: dict[int, str] = {}
    for index, match in enumerate(matches):
        turn = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        prompt = text[start:end].strip()
        if prompt:
            prompts[turn] = prompt
    return prompts


def load_prompts_by_turn(prompt_path: Path | None) -> dict[int, str]:
    if prompt_path is None or not prompt_path.is_file():
        return {}
    return parse_prompts_file(prompt_path.read_text(encoding="utf-8", errors="replace"))


def write_step_artifacts(
    artifacts_dir: Path,
    run_number: int,
    *,
    diff_patch: str,
    jsonl_text: str,
    agent_jsonl_name: str = "codex.jsonl",
    prompter_jsonl: str = "",
) -> None:
    run_dir = artifacts_dir / f"run_{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diff.patch").write_text(diff_patch, encoding="utf-8")
    (run_dir / agent_jsonl_name).write_text(jsonl_text, encoding="utf-8")
    if prompter_jsonl:
        (run_dir / "prompter.jsonl").write_text(prompter_jsonl, encoding="utf-8")
