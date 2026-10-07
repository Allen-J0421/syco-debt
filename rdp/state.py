"""Per-instance run state. One small JSON file per instance, replaced atomically.

There is no shared state file: overall progress is derived by scanning these files, so
parallel workers never contend on state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdp.util import append_jsonl_locked, atomic_write_json, read_json, utc_iso

PENDING, RUNNING, DONE, FAILED, BLOCKED = "pending", "running", "done", "failed", "blocked"


@dataclass
class InstanceState:
    instance_id: str
    status: str = PENDING
    attempts: int = 0            # runs started
    requeues: int = 0            # runs discarded by a provider limit
    stamp: str = ""
    branch: str = ""
    baseline_sha: str = ""
    turns_done: int = 0
    agent_failures: int = 0
    prompter_fallbacks: int = 0
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    discarded: list = field(default_factory=list)   # [{stamp, branch, reason}]


def instance_dir(run_dir: Path, instance_id: str) -> Path:
    return run_dir / "instances" / instance_id


def state_path(run_dir: Path, instance_id: str) -> Path:
    return instance_dir(run_dir, instance_id) / "state.json"


def load_state(run_dir: Path, instance_id: str) -> InstanceState:
    data = read_json(state_path(run_dir, instance_id))
    if not isinstance(data, dict):
        return InstanceState(instance_id=instance_id)
    known = InstanceState.__dataclass_fields__
    return InstanceState(**{k: v for k, v in data.items() if k in known})


def save_state(run_dir: Path, state: InstanceState) -> None:
    atomic_write_json(state_path(run_dir, state.instance_id), asdict(state))


def log_event(run_dir: Path, event: str, **fields) -> None:
    append_jsonl_locked(run_dir / "events.jsonl", {"ts": utc_iso(), "event": event, **fields})
