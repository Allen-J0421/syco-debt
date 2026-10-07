"""Git operations inside one instance's worktree."""

from __future__ import annotations

from pathlib import Path

from rdp.runner.models import LineStats
from rdp.util import run_text_command


def parse_numstat(numstat_text: str) -> dict[str, LineStats]:
    """``git diff --numstat`` output -> {path: stats}. Binary files count as changed, 0 lines."""
    out: dict[str, LineStats] = {}
    for line in numstat_text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        if added == "-" or deleted == "-":
            out[path] = LineStats(files_changed=1, lines_added=0, lines_deleted=0)
        else:
            out[path] = LineStats(files_changed=1, lines_added=int(added), lines_deleted=int(deleted))
    return out


def sum_stats(per_path: dict[str, LineStats], keep=None) -> LineStats:
    total = LineStats.empty()
    for path, stats in per_path.items():
        if keep is None or keep(path):
            total = total.plus(stats)
    return total


class GitRepository:
    """Git subprocess wrapper scoped to one worktree."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def run_git(self, args: list[str], *, check: bool = True):
        proc = run_text_command(["git", *args], cwd=self.repo_root)
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stderr}")
        return proc

    def head_sha(self) -> str:
        return self.run_git(["rev-parse", "HEAD"]).stdout.strip()

    def tree_sha(self, rev: str = "HEAD") -> str:
        return self.run_git(["rev-parse", f"{rev}^{{tree}}"]).stdout.strip()

    def stage_all(self) -> None:
        self.run_git(["add", "-A"])

    def staged_numstat(self, previous_sha: str) -> dict[str, LineStats]:
        """Per-file stats of the staged tree against ``previous_sha`` (renames not folded)."""
        out = self.run_git(["diff", "--cached", "--numstat", "--no-renames", previous_sha]).stdout
        return parse_numstat(out)

    def staged_diff(self, previous_sha: str) -> str:
        proc = self.run_git(["diff", "--cached", previous_sha], check=False)
        return proc.stdout or ""

    def unstage_and_restore(self, previous_sha: str) -> None:
        """Drop everything since ``previous_sha`` (used when a turn is discarded before commit)."""
        self.run_git(["reset", "--hard", previous_sha], check=False)
        self.run_git(["clean", "-fdq"], check=False)

    def commit(self, message: str) -> str:
        self.run_git(["commit", "-m", message, "--allow-empty", "--no-verify"])
        return self.head_sha()

    def file_text(self, commit: str, path: str) -> str:
        proc = run_text_command(["git", "show", f"{commit}:{path}"], cwd=self.repo_root)
        return proc.stdout if proc.returncode == 0 else ""
