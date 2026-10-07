"""Build the result tables from per-instance artifacts and the metrics cache."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from rdp.config import Settings
from rdp.instances import INSTANCE_COLUMNS, Instance
from rdp.metrics.aggregate import METRIC_COLS
from rdp.metrics.measure import load_cached
from rdp.metrics.scope import HUMAN, load_turn_records
from rdp.runner.claude_parse import extract_final_agent_message
from rdp.signals.compute import (
    StopClaimDetector,
    instance_signals,
    read_log_csv,
    turn_signals,
)
from rdp.state import DONE, instance_dir, load_state
from rdp.util import read_json, read_jsonl, sanitize_slug

# Same identity and metric columns as the foundational data/metrics.csv, then the additions.
ID_COLS = ["unit", "unit_type", "tool", "model", "timestamp", "branch", "iteration", "commit_sha", "tree_sha"]
EXTRA_COLS = ["arm", "project", "refactoring_type", "scope_files", "scope_hash"]
METRICS_COLUMNS = ID_COLS + METRIC_COLS + EXTRA_COLS

STEP_COLUMNS = [
    "instance_id", "turn", "commit_sha", "tree_sha",
    "files_changed", "lines_added", "lines_deleted", "lines_total",
    "java_files_changed", "java_lines_added", "java_lines_deleted", "java_lines_total",
    "target_files_changed", "target_lines_added", "target_lines_deleted", "target_lines_total",
    "stop", "stop_claim", "claim_categories", "refusal_edit", "post_stop", "post_stop_edit",
    "clarification_followup", "prompt_source", "prompter_fallback",
    "duration_s", "exit_code", "timed_out",
    "usage_agent_calls", "usage_agent_turns", "usage_input_tokens", "usage_output_tokens",
    "usage_cache_creation_input_tokens", "usage_cache_read_input_tokens", "usage_cost_usd",
    "usage_api_duration_ms",
]
SIGNAL_COLUMNS = [
    "instance_id", "project", "refactoring_type", "model", "stamp", "num_turns", "t_star",
    "never_stopped", "stop_turns", "relapsed", "post_stop_edit_turns", "pre_stop_lines",
    "post_stop_lines", "baseline_ncloc", "s2_post_stop", "stop_claim_turns", "refusal_edit_turns",
    "s3_refusal_edit_rate", "failed_turns", "turns_without_transcript", "prompter_fallbacks", "clean",
]
ISSUE_COLUMNS = ["instance_id", "version", "rule", "severity", "effort_min", "file", "line", "message"]
VERIFY_COLUMNS = ["instance_id", "version", "commit", "jdk", "outcome", "returncode", "duration_s", "command"]


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class RunData:
    """Everything recorded for one finished instance."""

    def __init__(self, settings: Settings, run_dir: Path, inst: Instance, detector: StopClaimDetector) -> None:
        self.inst = inst
        self.state = load_state(run_dir, inst.instance_id)
        self.dir = instance_dir(run_dir, inst.instance_id)
        self.model = str(settings.agent.get("model"))
        self.run_stamp = f"{self.state.stamp}-{sanitize_slug(self.model)}"
        self.ok = self.state.status == DONE
        self.turns: list[dict] = []
        self.signals = []
        if self.ok:
            self.turns = load_turn_records(self.dir / "logs" / f"{self.run_stamp}-turns.jsonl")
            log = self.dir / "logs" / f"{self.run_stamp}-log.csv"
            if log.is_file():
                self.signals = turn_signals(read_log_csv(log), self.dir / self.run_stamp, self.model, detector)
        self.snapshots = read_jsonl(self.dir / "snapshots.jsonl")
        self.scope = (read_json(self.dir / "scope.json", {}) or {}).get("files", [])


def assemble(settings: Settings, run_dir: Path, instances: list[Instance]) -> dict[str, int]:
    tables = run_dir / "tables"
    detector = StopClaimDetector()
    metrics_rows, step_rows, signal_rows, issue_rows, verify_rows = [], [], [], [], []

    for inst in instances:
        data = RunData(settings, run_dir, inst, detector)
        if not data.ok:
            continue
        baseline_ncloc = None

        for snap in data.snapshots:
            metrics, issues = load_cached(settings, snap["scope_hash"])
            is_human = snap["version"] == HUMAN
            if snap["version"] == "0":
                try:
                    baseline_ncloc = float(metrics.get("ncloc"))
                except (TypeError, ValueError):
                    baseline_ncloc = None
            metrics_rows.append({
                "unit": inst.instance_id, "unit_type": "SWE", "tool": "claude", "model": data.model,
                "timestamp": data.state.stamp, "branch": data.state.branch,
                "iteration": "" if is_human else snap["version"],
                "commit_sha": inst.commit_id if is_human else snap["commit"],
                "tree_sha": snap["tree_sha"],
                **{c: metrics.get(c) for c in METRIC_COLS},
                "arm": "human" if is_human else "agent", "project": inst.project,
                "refactoring_type": inst.refactoring_type, "scope_files": snap["n_files"],
                "scope_hash": snap["scope_hash"],
            })
            for issue in issues:
                issue_rows.append({"instance_id": inst.instance_id, "version": snap["version"], **issue})

        by_turn = {s.turn: s for s in data.signals}
        for rec in data.turns:
            sig = by_turn.get(int(rec["turn"]))
            row = dict(rec)
            if sig is not None:
                row.update(stop=int(sig.stop), stop_claim=int(sig.stop_claim),
                           claim_categories="|".join(sig.claim_categories),
                           refusal_edit=int(sig.refusal_edit), post_stop=int(sig.post_stop),
                           post_stop_edit=int(sig.post_stop_edit))
            step_rows.append(row)

        if data.signals:
            summary = instance_signals(data.signals, baseline_lines=baseline_ncloc)
            signal_rows.append({
                "instance_id": inst.instance_id, "project": inst.project,
                "refactoring_type": inst.refactoring_type, "model": data.model,
                "stamp": data.state.stamp, "baseline_ncloc": baseline_ncloc,
                "prompter_fallbacks": data.state.prompter_fallbacks, **summary,
            })
            if data.state.prompter_fallbacks:
                signal_rows[-1]["clean"] = 0   # a fallback prompt changed the treatment

        verify_rows.extend(read_jsonl(data.dir / "verify.jsonl"))

    metrics_rows.sort(key=lambda r: (r["unit"], r["arm"], int(r["iteration"] or 0)))
    _write(tables / "metrics.csv", METRICS_COLUMNS, metrics_rows)
    _write(tables / "steps.csv", STEP_COLUMNS, step_rows)
    _write(tables / "signals.csv", SIGNAL_COLUMNS, signal_rows)
    _write(tables / "issues.csv", ISSUE_COLUMNS, issue_rows)
    _write(tables / "verify.csv", VERIFY_COLUMNS, verify_rows)
    _write(tables / "instances.csv", INSTANCE_COLUMNS, [i.to_json() for i in instances])
    return {"metrics": len(metrics_rows), "steps": len(step_rows), "signals": len(signal_rows),
            "issues": len(issue_rows), "verify": len(verify_rows), "instances": len(instances)}


LABEL_COLUMNS = ["instance_id", "turn", "lines_changed", "detector_stop_claim", "detector_categories",
                 "human_stop_claim", "notes", "final_message"]


def label_sample(settings: Settings, run_dir: Path, instances: list[Instance], *, n: int, seed: int) -> int:
    """A sheet for hand-checking the stop-claim detector, balanced over (claim, edit) cells."""
    detector = StopClaimDetector()
    cells: dict[tuple[bool, bool], list[dict]] = {}
    for inst in instances:
        data = RunData(settings, run_dir, inst, detector)
        for sig in data.signals:
            jsonl = data.dir / data.run_stamp / f"run_{sig.turn:03d}" / "claude.jsonl"
            text = extract_final_agent_message(jsonl.read_text(errors="replace")) if jsonl.is_file() else ""
            cells.setdefault((sig.stop_claim, sig.lines_changed > 0), []).append({
                "instance_id": inst.instance_id, "turn": sig.turn, "lines_changed": sig.lines_changed,
                "detector_stop_claim": int(sig.stop_claim),
                "detector_categories": "|".join(sig.claim_categories),
                "human_stop_claim": "", "notes": "", "final_message": text[:3000],
            })
    rng = random.Random(seed)
    for rows in cells.values():
        rng.shuffle(rows)
    picked: list[dict] = []
    while len(picked) < n and any(cells.values()):
        for key in sorted(cells):
            if cells[key] and len(picked) < n:
                picked.append(cells[key].pop())
    _write(run_dir / "tables" / "label_sample.csv", LABEL_COLUMNS, picked)
    return len(picked)
