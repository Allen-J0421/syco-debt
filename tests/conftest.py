from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rdp.instances import Instance  # noqa: E402
from rdp.repos import RepoCache  # noqa: E402

JAVA_V0 = """package demo;

public class Calc {
    public int sum(int a, int b) {
        int r = a + b;
        return r;
    }
}
"""
JAVA_HUMAN = """package demo;

public class Calc {
    public int sum(int a, int b) {
        return add(a, b);
    }

    private int add(int a, int b) {
        return a + b;
    }
}
"""
TARGET = "src/main/java/demo/Calc.java"


def git(cwd: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> dict:
    """A tiny upstream repo: commit A (before), commit B (the human refactoring)."""
    repo = tmp_path / "upstream"
    (repo / "src/main/java/demo").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    (repo / TARGET).write_text(JAVA_V0)
    (repo / "README.md").write_text("demo\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "before")
    before = git(repo, "rev-parse", "HEAD")
    (repo / TARGET).write_text(JAVA_HUMAN)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "human refactor")
    human = git(repo, "rev-parse", "HEAD")
    return {"path": repo, "url": str(repo), "before": before, "human": human}


@pytest.fixture
def repos(tmp_path: Path) -> RepoCache:
    return RepoCache(tmp_path / "home" / "repos", tmp_path / "home" / "worktrees")


def make_instance(upstream: dict, n: int, project: str = "demo") -> Instance:
    return Instance(
        instance_id=f"{project}-{n:02d}", unique_id=f"u{n}", project=project, repo_url=upstream["url"],
        commit_id=upstream["human"], refactoring_type="Extract Method", target_file=TARGET,
        file_after=TARGET, method="sum", method_qualified="demo.Calc#sum", human_files=(TARGET,),
        compile_jdk="17", compile_command="true", is_test_code=False,
    )


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch) -> Path:
    state = tmp_path / "fake_state"
    monkeypatch.setenv("RDP_CLAUDE_BIN", str(Path(__file__).parent / "fake_claude.py"))
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(state))
    return state
