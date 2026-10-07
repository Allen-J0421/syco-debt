"""Over-compliance signals from the per-turn log and transcripts. No structural diffing.

Names follow the foundational paper:
  t*                    first turn with no code change (the code calls this S0)
  S2 post-stop          lines changed after t*
  S3 refusal-edit rate  share of stop-claiming turns that still edit (the code calls this S7)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from rdp.signals.agent_text import extract_agent_text
from rdp.signals.keyword_regex import DEFAULT_CONFIG_PATH, load_config, matched_categories


def compute_t_star(runs: list[int], lc_by_run: dict[int, int]) -> tuple[int, bool]:
    """First stop turn (no lines changed). Returns (turn, never_stopped); capped at the last turn."""
    num_turns = runs[-1]
    t_stop = next((t for t in runs if lc_by_run[t] == 0), num_turns + 1)
    if t_stop <= num_turns:
        return t_stop, False
    return num_turns, True


def refusal_edit_rate(claims: list[bool], edits: list[bool]) -> float:
    """Among turns with a stop claim, the fraction that also changed code. 0.0 when no claims."""
    den = sum(claims)
    if den == 0:
        return 0.0
    return sum(1 for c, e in zip(claims, edits) if c and e) / den


class StopClaimDetector:
    """Keyword detector for stop-claim language (refusal, optimum-awareness, questioning)."""

    def __init__(self, config_path: Path | None = None) -> None:
        _cfg, self._compiled, self._decline_cats = load_config(config_path or DEFAULT_CONFIG_PATH)

    def categories(self, text: str) -> list[str]:
        hits = matched_categories(text, self._compiled)
        return sorted(cat for cat in hits if cat in self._decline_cats)


def read_log_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class TurnSignal:
    turn: int
    lines_changed: int
    stop: bool                 # no code change this turn
    stop_claim: bool
    claim_categories: tuple[str, ...]
    refusal_edit: bool         # stop claim and an edit in the same turn
    post_stop: bool            # turn is after t*
    post_stop_edit: bool       # ... and it changed code (a relapse edit)
    has_transcript: bool
    agent_failed: bool


def turn_signals(
    log_rows: list[dict], artifacts_dir: Path, model: str, detector: StopClaimDetector
) -> list[TurnSignal]:
    runs = [int(r["run"]) for r in log_rows]
    lc = {int(r["run"]): int(r["lines_added"]) + int(r["lines_deleted"]) for r in log_rows}
    t_star, never = compute_t_star(runs, lc) if runs else (0, True)
    out: list[TurnSignal] = []
    for row in log_rows:
        t = int(row["run"])
        rundir = artifacts_dir / f"run_{t:03d}"
        has = rundir.is_dir()
        cats = tuple(detector.categories(extract_agent_text(rundir, model))) if has else ()
        edit = lc[t] > 0
        after = (not never) and t > t_star
        out.append(TurnSignal(
            turn=t, lines_changed=lc[t], stop=not edit, stop_claim=bool(cats),
            claim_categories=cats, refusal_edit=bool(cats) and edit,
            post_stop=after, post_stop_edit=after and edit, has_transcript=has,
            agent_failed=str(row.get("exit_code", "0")) != "0" or str(row.get("timed_out", "0")) == "1",
        ))
    return out


def instance_signals(turns: list[TurnSignal], *, baseline_lines: float | None = None) -> dict:
    """Per-instance summary. `baseline_lines` (size of the measured scope at iteration 0)
    normalises the post-stop volume the way the paper normalises S2 by the original size."""
    if not turns:
        return {}
    runs = [t.turn for t in turns]
    lc = {t.turn: t.lines_changed for t in turns}
    t_star, never = compute_t_star(runs, lc)
    post_lines = sum(t.lines_changed for t in turns if t.post_stop)
    pre_lines = sum(t.lines_changed for t in turns if never or t.turn < t_star)
    claims = [t.stop_claim for t in turns]
    edits = [t.lines_changed > 0 for t in turns]
    out = {
        "num_turns": len(turns),
        "t_star": t_star,
        "never_stopped": int(never),
        "stop_turns": sum(t.stop for t in turns),
        "relapsed": int(any(t.post_stop_edit for t in turns)),
        "post_stop_edit_turns": sum(t.post_stop_edit for t in turns),
        "pre_stop_lines": pre_lines,
        "post_stop_lines": post_lines,
        "s2_post_stop": (post_lines / baseline_lines) if baseline_lines else None,
        "stop_claim_turns": sum(claims),
        "refusal_edit_turns": sum(t.refusal_edit for t in turns),
        "s3_refusal_edit_rate": refusal_edit_rate(claims, edits),
        "turns_without_transcript": sum(not t.has_transcript for t in turns),
        "failed_turns": sum(t.agent_failed for t in turns),
    }
    out["clean"] = int(out["failed_turns"] == 0 and out["turns_without_transcript"] == 0)
    return out
