"""Bare mirrors (one per project) and per-instance worktrees.

Parallel safety: every instance works in its own worktree on its own branch, so concurrent
commits never touch the same ref. Only worktree add/remove/prune share metadata in the mirror
and are serialised by a per-project lock. Background gc is disabled in the mirrors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rdp.util import file_lock, run_text_command


class GitError(RuntimeError):
    pass


def _git(args: list[str], *, cwd: Path | None = None, check: bool = True, timeout: int | None = None):
    proc = run_text_command(["git", *args], cwd=cwd, timeout=timeout)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {cwd}:\n{proc.stderr.strip()}")
    return proc


class RepoCache:
    def __init__(self, repos_dir: Path, worktrees_dir: Path) -> None:
        self.repos_dir = repos_dir
        self.worktrees_dir = worktrees_dir

    # ---- mirrors ----------------------------------------------------------------
    def mirror_path(self, project: str) -> Path:
        return self.repos_dir / f"{project}.git"

    def _lock(self, project: str):
        return file_lock(self.repos_dir / f".{project}.lock")

    def ensure_mirror(self, project: str, url: str, *, commits: list[str] | None = None) -> Path:
        """Clone the project once (bare), then make sure the needed commits are present."""
        path = self.mirror_path(project)
        with self._lock(project):
            if not (path / "HEAD").is_file():
                self.repos_dir.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".partial")
                shutil.rmtree(tmp, ignore_errors=True)
                _git(["clone", "--bare", "--quiet", url, str(tmp)])
                tmp.rename(path)
                _git(["config", "gc.auto", "0"], cwd=path)
                _git(["config", "maintenance.auto", "false"], cwd=path, check=False)
            missing = [c for c in (commits or []) if not self.has_commit(project, c)]
            if missing:
                _git(["fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*", "--tags"],
                     cwd=path, check=False)
                for c in [c for c in missing if not self.has_commit(project, c)]:
                    _git(["fetch", "--quiet", "origin", c], cwd=path, check=False)
        return path

    def has_commit(self, project: str, commit: str) -> bool:
        proc = _git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=self.mirror_path(project), check=False)
        return proc.returncode == 0

    def rev_parse(self, project: str, rev: str) -> str:
        return _git(["rev-parse", rev], cwd=self.mirror_path(project)).stdout.strip()

    def parent_of(self, project: str, commit: str) -> str:
        return self.rev_parse(project, f"{commit}^")

    # ---- worktrees --------------------------------------------------------------
    def worktree_path(self, instance_id: str) -> Path:
        return self.worktrees_dir / instance_id

    def add_worktree(self, project: str, instance_id: str, start_commit: str, branch: str) -> Path:
        """Fresh worktree at start_commit on a new branch. Replaces any leftover of this instance."""
        mirror = self.mirror_path(project)
        wt = self.worktree_path(instance_id)
        with self._lock(project):
            self._remove_worktree_locked(mirror, wt)
            self.worktrees_dir.mkdir(parents=True, exist_ok=True)
            _git(["worktree", "add", "--quiet", "-b", branch, str(wt), start_commit], cwd=mirror)
        _git(["config", "user.name", "rdp"], cwd=wt, check=False)
        _git(["config", "user.email", "rdp@localhost"], cwd=wt, check=False)
        return wt

    def _remove_worktree_locked(self, mirror: Path, wt: Path) -> None:
        if wt.exists():
            _git(["worktree", "remove", "--force", str(wt)], cwd=mirror, check=False)
            shutil.rmtree(wt, ignore_errors=True)
        _git(["worktree", "prune"], cwd=mirror, check=False)

    def remove_worktree(self, project: str, instance_id: str) -> None:
        with self._lock(project):
            self._remove_worktree_locked(self.mirror_path(project), self.worktree_path(instance_id))

    def delete_branch(self, project: str, branch: str) -> None:
        with self._lock(project):
            _git(["branch", "-D", branch], cwd=self.mirror_path(project), check=False)

    def branch_exists(self, project: str, branch: str) -> bool:
        proc = _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
                    cwd=self.mirror_path(project), check=False)
        return proc.returncode == 0

    # ---- read-only snapshot access (no worktree needed) --------------------------
    def tree_sha(self, project: str, commit: str) -> str:
        return self.rev_parse(project, f"{commit}^{{tree}}")

    def show_file(self, project: str, commit: str, path: str) -> bytes | None:
        """File content at a commit, or None when the file does not exist there."""
        import subprocess

        proc = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=self.mirror_path(project),
                              capture_output=True)
        return proc.stdout if proc.returncode == 0 else None

    def changed_files(self, project: str, old: str, new: str) -> list[str]:
        out = _git(["diff", "--name-only", "--no-renames", old, new], cwd=self.mirror_path(project)).stdout
        return [line for line in out.splitlines() if line.strip()]

    def export_tree(self, project: str, commit: str, dest: Path) -> None:
        """Materialise a commit's full tree into dest (git archive | tar)."""
        import subprocess

        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        archive = subprocess.Popen(["git", "archive", commit], cwd=self.mirror_path(project),
                                   stdout=subprocess.PIPE)
        tar = subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout)
        archive.stdout.close()
        if archive.wait() != 0 or tar.returncode != 0:
            raise GitError(f"git archive {commit} failed for {project}")
