"""Bare mirrors (one per project) and an isolated workspace per instance.

Isolation: the agent works in a standalone repository that holds a single commit with the
files of the starting point. It has no project history, no remotes and no other branches, so
the human refactoring commit, later upstream history and other instances' runs cannot be read
from git. When a run finishes, its branch is fetched into the mirror, where later stages read
every version by commit.

Parallel safety: workspaces are separate repositories, so running instances share nothing.
Only the final fetch into a mirror is serialised, by a per-project lock. Background gc is
disabled in the mirrors.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rdp.util import file_lock, run_text_command


class GitError(RuntimeError):
    pass


def _git(args: list[str], *, cwd: Path | None = None, check: bool = True, timeout: int | None = None,
         env: dict[str, str] | None = None):
    proc = run_text_command(["git", *args], cwd=cwd, timeout=timeout, env=env)
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

    # ---- isolated workspaces ----------------------------------------------------
    def workspace_path(self, instance_id: str, project: str) -> Path:
        """The agent's working directory. Named after the project, inside a per-instance folder."""
        return self.worktrees_dir / instance_id / project

    def create_workspace(
        self, project: str, instance_id: str, start_commit: str, branch: str
    ) -> tuple[Path, str, bool]:
        """Standalone repository whose only commit holds the files of `start_commit`.

        Returns (path, baseline commit, whether the baseline tree equals the upstream tree).
        Any leftover workspace of this instance is replaced."""
        mirror = self.mirror_path(project)
        ws = self.workspace_path(instance_id, project)
        self.remove_workspace(instance_id)
        ws.mkdir(parents=True)
        # Check the files out of the mirror with a private index: nothing in the mirror is written.
        index = ws.parent / ".checkout-index"
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        base = ["--git-dir", str(mirror), "--work-tree", str(ws)]
        try:
            _git([*base, "read-tree", start_commit], env=env)
            _git([*base, "checkout-index", "-a", "-f"], env=env)
        finally:
            index.unlink(missing_ok=True)
        _git(["init", "--quiet", "-b", branch], cwd=ws)
        _git(["config", "user.name", "rdp"], cwd=ws)
        _git(["config", "user.email", "rdp@localhost"], cwd=ws)
        _git(["config", "gc.auto", "0"], cwd=ws)
        _git(["add", "-A", "-f"], cwd=ws)          # -f: also files the project itself ignores but tracks
        _git(["commit", "--quiet", "--no-verify", "--allow-empty", "-m", "baseline"], cwd=ws)
        baseline = _git(["rev-parse", "HEAD"], cwd=ws).stdout.strip()
        tree = _git(["rev-parse", "HEAD^{tree}"], cwd=ws).stdout.strip()
        return ws, baseline, tree == self.tree_sha(project, start_commit)

    def publish_branch(self, project: str, instance_id: str, branch: str) -> None:
        """Copy the finished run's branch from its workspace into the mirror."""
        ws = self.workspace_path(instance_id, project)
        with self._lock(project):
            _git(["fetch", "--quiet", "--no-tags", str(ws), f"+refs/heads/{branch}:refs/heads/{branch}"],
                 cwd=self.mirror_path(project))

    def remove_workspace(self, instance_id: str) -> None:
        shutil.rmtree(self.worktrees_dir / instance_id, ignore_errors=True)

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
