"""Claude Code CLI agent wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from rdp.runner.models import AgentRunResult
from rdp.util import run_text_command

CLAUDE_AUTO_FLAGS = ("--verbose", "--permission-mode", "bypassPermissions")
DEFAULT_TIMEOUT = 2700


def claude_binary() -> str:
    """The Claude Code CLI to run. RDP_CLAUDE_BIN overrides it (tests use a fake)."""
    return os.environ.get("RDP_CLAUDE_BIN") or "claude"


def build_claude_command(
    prompt: str,
    model: str,
    *,
    is_first: bool,
    session_id: str | None = None,
    effort: str | None = None,
) -> list[str]:
    cmd = [
        claude_binary(),
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        *CLAUDE_AUTO_FLAGS,
        "--model",
        model,
    ]
    if effort:
        cmd.extend(["--effort", effort])
    if not is_first:
        if not session_id:
            raise RuntimeError("Missing Claude session id for resume.")
        cmd.extend(["--resume", session_id])
    return cmd


def parse_claude_session_id(jsonl_text: str) -> str | None:
    """
    Extract the Claude resume id from ``claude -p --output-format stream-json`` output.

    Prefer ``type==result`` or ``type==system`` + ``subtype==init``; fall back to the
    last non-empty ``session_id`` on any JSON line.
    """
    preferred: str | None = None
    fallback: str | None = None
    for raw in jsonl_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        sid = obj.get("session_id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        sid = sid.strip()
        fallback = sid
        event_type = obj.get("type")
        if event_type == "result":
            preferred = sid
        elif event_type == "system" and obj.get("subtype") == "init":
            preferred = sid
    return preferred or fallback


class ClaudeAgent:
    """Runs ``claude -p`` / ``claude -p --resume`` for one working directory."""

    def __init__(
        self,
        work_dir: Path,
        model: str,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        effort: str | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.model = model
        self.timeout = timeout
        self.effort = effort

    def run(
        self,
        prompt: str,
        *,
        is_first: bool,
        session_id: str | None = None,
    ) -> tuple[AgentRunResult, str]:
        cmd = build_claude_command(
            prompt,
            self.model,
            is_first=is_first,
            session_id=session_id,
            effort=self.effort,
        )
        t0 = time.monotonic()
        try:
            proc = run_text_command(cmd, cwd=self.work_dir, timeout=self.timeout)
            out = proc.stdout or ""
            err = proc.stderr or ""
            jsonl = "\n".join([out, err])
            sid: str | None = None
            if is_first:
                sid = parse_claude_session_id(jsonl)
                if not sid:
                    preview: list[str] = []
                    if out.strip():
                        preview.append(
                            "stdout (first 10 lines):\n" + "\n".join(out.splitlines()[:10])
                        )
                    if err.strip():
                        preview.append(
                            "stderr (first 10 lines):\n" + "\n".join(err.splitlines()[:10])
                        )
                    detail = ("\n\n" + "\n\n".join(preview)) if preview else ""
                    raise RuntimeError(
                        "Could not parse Claude session id from "
                        "`claude -p --output-format stream-json` output. "
                        "Try running with a longer timeout, or run "
                        "`claude -p --output-format stream-json ...` "
                        f"manually to inspect output.{detail}"
                    )
            return (
                AgentRunResult(
                    exit_code=proc.returncode,
                    duration_s=time.monotonic() - t0,
                    timed_out=False,
                    session_id=sid,
                ),
                jsonl,
            )
        except subprocess.TimeoutExpired:
            return (
                AgentRunResult(
                    exit_code=124,
                    duration_s=time.monotonic() - t0,
                    timed_out=True,
                ),
                "",
            )
