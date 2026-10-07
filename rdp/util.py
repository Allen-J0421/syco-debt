"""Small shared utilities."""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def run_text_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        env=env,
    )


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_slug(s: str, *, max_len: int = 64) -> str:
    """Safe fragment for git branch paths and filenames."""
    t = (s or "").strip().replace("\\", "-").replace("/", "-").replace(" ", "_")
    t = re.sub(r"[^a-zA-Z0-9._-]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-_.")
    if not t:
        t = "default"
    return t[:max_len]


def non_empty_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock across threads (in-process lock) and processes (flock)."""
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        tlock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    path.parent.mkdir(parents=True, exist_ok=True)
    with tlock:
        with open(path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def append_jsonl_locked(path: Path, obj: dict) -> None:
    """Append one JSON line to a log shared by several workers."""
    with file_lock(path.with_name(path.name + ".lock")):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
