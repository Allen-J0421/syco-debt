"""Domain models for one instance's multi-turn run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LineStats:
    files_changed: int
    lines_added: int
    lines_deleted: int

    @classmethod
    def empty(cls) -> LineStats:
        return cls(files_changed=0, lines_added=0, lines_deleted=0)

    @property
    def lines_total(self) -> int:
        return self.lines_added + self.lines_deleted

    def plus(self, other: LineStats) -> LineStats:
        return LineStats(
            files_changed=self.files_changed + other.files_changed,
            lines_added=self.lines_added + other.lines_added,
            lines_deleted=self.lines_deleted + other.lines_deleted,
        )

    def as_dict(self, prefix: str = "") -> dict[str, int]:
        return {
            f"{prefix}files_changed": self.files_changed,
            f"{prefix}lines_added": self.lines_added,
            f"{prefix}lines_deleted": self.lines_deleted,
            f"{prefix}lines_total": self.lines_total,
        }


@dataclass(frozen=True)
class AgentRunResult:
    exit_code: int
    duration_s: float
    timed_out: bool
    session_id: str | None = None


def agent_run_ok(agent: AgentRunResult) -> bool:
    """True only if the coding agent exited 0 and did not time out."""
    return agent.exit_code == 0 and not agent.timed_out


@dataclass(frozen=True)
class ClarificationPatterns:
    refusal_blockers: tuple[re.Pattern[str], ...]
    clarification_signals: tuple[re.Pattern[str], ...]
    require_question_mark: bool = True


@dataclass(frozen=True)
class PrompterConfig:
    api_key: str
    model: str
    system_prompt: str
    nudge: str
    fallback_prompt: str
    clarification_nudge: str = "The coding agent asked for clarification above."
    retries: int = 3


@dataclass(frozen=True)
class PromptTurn:
    prompt: str
    prompter_jsonl: str = ""
    fallback: bool = False      # the prompter failed and the fixed fallback prompt was used
    source: str = "prompter"    # "template" (turn 1) | "prompter" | "fixed"


@dataclass(frozen=True)
class PrompterTurnInput:
    """Context passed from the loop into the simulated user each turn."""

    coding_reply: str | None = None
    diff_patch: str | None = None


@dataclass(frozen=True)
class IterationResult:
    number: int
    stats: LineStats                 # whole repository (defines a stop)
    java_stats: LineStats            # .java files only
    target_stats: LineStats          # the instance's target file only
    agent: AgentRunResult
    commit_sha: str
    commit_message: str
    tree_sha: str = ""
    jsonl_text: str = ""
    diff_patch: str = ""
    changed_paths: tuple[str, ...] = ()
    clarification_prompt_text: str = ""
    clarification_followup: bool = False
    clarification_prompter_jsonl: str = ""
    clarification_fallback: bool = False
    usage: dict = field(default_factory=dict)

    def as_csv_row(self, *, model: str, branch: str) -> list[object]:
        return [
            self.number,
            self.stats.files_changed,
            self.stats.lines_added,
            self.stats.lines_deleted,
            self.stats.lines_total,
            f"{self.agent.duration_s:.3f}",
            self.agent.exit_code,
            int(self.agent.timed_out),
            self.commit_sha,
            self.commit_message,
            model,
            branch,
        ]


# Same columns as the foundational harness's per-run log, so its tooling still reads these files.
CSV_COLUMNS = (
    "run", "files_changed", "lines_added", "lines_deleted", "lines_total", "duration_s",
    "exit_code", "timed_out", "commit_sha", "commit_message", "model", "git_branch",
)
