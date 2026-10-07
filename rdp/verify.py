"""Compile and test selected versions of each instance with the dataset's own build command.

Checked versions: 0 (baseline, as an environment sanity check), 1, the first stop (if any),
and the last turn. Results are appended per instance, so the stage resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdp.config import Settings
from rdp.instances import Instance
from rdp.metrics.scope import load_turn_records, version_commits
from rdp.repos import RepoCache
from rdp.signals.compute import compute_t_star
from rdp.state import DONE, instance_dir, load_state
from rdp.util import eprint, read_jsonl, sanitize_slug

_COMPILE_RE = re.compile(r"COMPILATION ERROR|Compilation failure|compileJava FAILED|compileTestJava FAILED"
                         r"|error: cannot find symbol", re.I)
_TEST_RE = re.compile(r"There are test failures|Tests run:.*(Failures: [1-9]|Errors: [1-9])"
                      r"|There were failing tests|> Task :\S*test FAILED", re.I)


def classify(returncode: int, timed_out: bool, log_text: str) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "pass"
    if _COMPILE_RE.search(log_text):
        return "compile_error"
    if _TEST_RE.search(log_text):
        return "test_failure"
    return "build_error"


def versions_to_check(turns: list[dict]) -> list[int]:
    runs = [int(t["turn"]) for t in turns]
    lc = {int(t["turn"]): int(t["lines_total"]) for t in turns}
    picks = {0, 1, runs[-1]}
    t_star, never = compute_t_star(runs, lc)
    if not never:
        picks.add(t_star)
    return sorted(picks)


def verify_instance(settings: Settings, repos: RepoCache, run_dir: Path, inst: Instance) -> list[dict]:
    state = load_state(run_dir, inst.instance_id)
    if state.status != DONE:
        return []
    idir = instance_dir(run_dir, inst.instance_id)
    out_path = idir / "verify.jsonl"
    done = {int(r["version"]) for r in read_jsonl(out_path)}
    model_slug = sanitize_slug(str(settings.agent.get("model")))
    turns = load_turn_records(idir / "logs" / f"{state.stamp}-{model_slug}-turns.jsonl")
    if not turns:
        return []
    commits = version_commits(state.baseline_sha, turns)
    cfg = settings.verify
    jdk_home = (cfg.get("jdk_homes") or {}).get(inst.compile_jdk, "")
    timeout = int(cfg.get("timeout_s", 3600))
    rows: list[dict] = []
    for version in versions_to_check(turns):
        if version in done:
            continue
        row = {"instance_id": inst.instance_id, "version": version, "commit": commits[version],
               "jdk": inst.compile_jdk, "command": inst.compile_command}
        if not inst.compile_command.strip():
            row.update(outcome="no_command", duration_s=0, log_tail="")
        elif not jdk_home or not (Path(jdk_home).expanduser() / "bin" / "java").is_file():
            row.update(outcome="no_jdk", duration_s=0, log_tail=f"no JDK {inst.compile_jdk} at {jdk_home!r}")
        else:
            work = settings.scratch_dir / f"v_{inst.instance_id}_{version}"
            try:
                repos.export_tree(inst.project, commits[version], work)
                jdk = str(Path(jdk_home).expanduser())
                env = dict(os.environ, JAVA_HOME=jdk, PATH=f"{jdk}/bin:{os.environ.get('PATH', '')}")
                t0 = time.monotonic()
                timed_out = False
                try:
                    proc = subprocess.run(["bash", "-c", inst.compile_command], cwd=str(work), env=env,
                                          capture_output=True, text=True, errors="replace", timeout=timeout)
                    text, rc = (proc.stdout or "") + (proc.stderr or ""), proc.returncode
                except subprocess.TimeoutExpired as exc:
                    out = exc.stdout or b""
                    text, rc, timed_out = (out.decode(errors="replace") if isinstance(out, bytes) else out), 124, True
                row.update(outcome=classify(rc, timed_out, text), returncode=rc,
                           duration_s=round(time.monotonic() - t0, 1), log_tail=text[-1500:])
            except Exception as exc:
                row.update(outcome="build_error", duration_s=0, log_tail=repr(exc))
            finally:
                shutil.rmtree(work, ignore_errors=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        eprint(f"[verify] {inst.instance_id} v{version}: {row['outcome']}")
    return rows


def verify_run(settings: Settings, repos: RepoCache, run_dir: Path, instances: list[Instance],
               *, jobs: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for rows in pool.map(lambda i: verify_instance(settings, repos, run_dir, i), instances):
            for row in rows:
                counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return counts
