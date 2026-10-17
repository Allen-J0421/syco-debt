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
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdp.config import Settings
from rdp.instances import Instance
from rdp.metrics.scope import load_turn_records, version_commits
from rdp.repos import RepoCache
from rdp.signals.compute import compute_t_star
from rdp.state import DONE, instance_dir, load_state
from rdp.util import eprint, file_lock, read_jsonl, sanitize_slug

_COMPILE_RE = re.compile(r"COMPILATION ERROR|Compilation failure|compileJava FAILED|compileTestJava FAILED"
                         r"|error: cannot find symbol", re.I)
_TEST_RE = re.compile(r"There are test failures|Tests run:.*(Failures: [1-9]|Errors: [1-9])"
                      r"|There were failing tests|> Task :\S*test FAILED", re.I)


def classify(returncode: int, timed_out: bool, log_text: str) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "pass"
    if returncode == 127 and "command not found" in log_text:
        return "missing_tool"
    if returncode == 2 and "unexpected EOF while looking for matching" in log_text:
        return "invalid_command"
    # A Gradle task named compileJava can fail while loading a corrupt remote
    # cache entry; that is a build-service failure, not a Java compiler error.
    if "Could not load from remote cache" in log_text:
        return "build_error"
    if _COMPILE_RE.search(log_text):
        return "compile_error"
    if _TEST_RE.search(log_text):
        return "test_failure"
    return "build_error"


def is_infrastructure_result(row: dict) -> bool:
    """Old build_error rows may contain setup failures recorded before classification improved."""
    outcome = row.get("outcome")
    if outcome in {"no_jdk", "no_command", "missing_tool", "invalid_command"}:
        return True
    if outcome != "build_error":
        return False
    return classify(int(row.get("returncode") or 1), False, str(row.get("log_tail") or "")) in {
        "missing_tool", "invalid_command",
    }


def versions_to_check(turns: list[dict]) -> list[int]:
    runs = [int(t["turn"]) for t in turns]
    lc = {int(t["turn"]): int(t["lines_total"]) for t in turns}
    picks = {0, 1, runs[-1]}
    t_star, never = compute_t_star(runs, lc)
    if not never:
        picks.add(t_star)
    return sorted(picks)


def verify_instance(settings: Settings, repos: RepoCache, run_dir: Path, inst: Instance,
                    *, retry_infra: bool = False, recheck: bool = False) -> list[dict]:
    # Keep the scratch export and append-only audit log in sync if two CLI
    # processes happen to select the same subject at once.
    with file_lock(instance_dir(run_dir, inst.instance_id) / ".verify.lock"):
        return _verify_instance_locked(settings, repos, run_dir, inst,
                                       retry_infra=retry_infra, recheck=recheck)


def _verify_instance_locked(settings: Settings, repos: RepoCache, run_dir: Path, inst: Instance,
                            *, retry_infra: bool, recheck: bool) -> list[dict]:
    state = load_state(run_dir, inst.instance_id)
    if state.status != DONE:
        return []
    idir = instance_dir(run_dir, inst.instance_id)
    out_path = idir / "verify.jsonl"
    history = read_jsonl(out_path)
    latest = {int(r["version"]): r for r in history}
    model_slug = sanitize_slug(str(settings.agent.get("model")))
    turns = load_turn_records(idir / "logs" / f"{state.stamp}-{model_slug}-turns.jsonl")
    if not turns:
        return []
    commits = version_commits(state.baseline_sha, turns)
    cfg = settings.verify
    jdk_home = (cfg.get("jdk_homes") or {}).get(inst.compile_jdk, "")
    maven_home = str(cfg.get("maven_home") or "")
    command = (cfg.get("command_overrides") or {}).get(inst.instance_id, inst.compile_command)
    timeout = int((cfg.get("timeout_overrides") or {}).get(inst.instance_id,
                                                        cfg.get("timeout_s", 3600)))
    rows: list[dict] = []
    for version in versions_to_check(turns):
        if version in latest and not (recheck or (retry_infra and is_infrastructure_result(latest[version]))):
            continue
        row = {"instance_id": inst.instance_id, "version": version, "commit": commits[version],
               "jdk": inst.compile_jdk, "command": command,
               "source_command": inst.compile_command,
               "attempt": sum(int(r["version"]) == version for r in history) + 1}
        if not command.strip():
            row.update(outcome="no_command", duration_s=0, log_tail="")
        elif not jdk_home or not (Path(jdk_home).expanduser() / "bin" / "java").is_file():
            row.update(outcome="no_jdk", duration_s=0, log_tail=f"no JDK {inst.compile_jdk} at {jdk_home!r}")
        else:
            work = settings.verify_scratch_dir / f"v_{inst.instance_id}_{version}"
            try:
                repos.export_tree(inst.project, commits[version], work)
                jdk = str(Path(jdk_home).expanduser())
                path_parts = [f"{jdk}/bin"]
                if maven_home:
                    path_parts.append(str(Path(maven_home).expanduser() / "bin"))
                path_parts.append(os.environ.get("PATH", ""))
                env = dict(os.environ, JAVA_HOME=jdk, PATH=":".join(path_parts))
                t0 = time.monotonic()
                timed_out = False
                try:
                    proc = subprocess.run(["bash", "-c", command], cwd=str(work), env=env,
                                          capture_output=True, text=True, errors="replace", timeout=timeout)
                    text, rc = (proc.stdout or "") + (proc.stderr or ""), proc.returncode
                except subprocess.TimeoutExpired as exc:
                    out = exc.stdout or b""
                    err = exc.stderr or b""
                    stdout = out.decode(errors="replace") if isinstance(out, bytes) else out
                    stderr = err.decode(errors="replace") if isinstance(err, bytes) else err
                    text, rc, timed_out = stdout + stderr, 124, True
                log_path = idir / "logs" / f"verify-v{version}-attempt{row['attempt']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(text, encoding="utf-8")
                row.update(outcome=classify(rc, timed_out, text), returncode=rc,
                           duration_s=round(time.monotonic() - t0, 1), log_tail=text[-1500:],
                           log_file=str(log_path.relative_to(idir)))
                if rc != 0:
                    report_files = [p for name in ("surefire-reports", "failsafe-reports")
                                    for report_dir in work.rglob(name)
                                    for p in report_dir.rglob("*") if p.is_file()]
                    if report_files:
                        archive = idir / "logs" / f"verify-v{version}-attempt{row['attempt']}-reports.zip"
                        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                            for p in report_files:
                                bundle.write(p, p.relative_to(work))
                        row["report_archive"] = str(archive.relative_to(idir))
            except Exception as exc:
                row.update(outcome="build_error", duration_s=0, log_tail=repr(exc))
            finally:
                shutil.rmtree(work, ignore_errors=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        history.append(row)
        latest[version] = row
        rows.append(row)
        eprint(f"[verify] {inst.instance_id} v{version}: {row['outcome']}")
    return rows


def verify_run(settings: Settings, repos: RepoCache, run_dir: Path, instances: list[Instance],
               *, jobs: int, retry_infra: bool = False, recheck: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for rows in pool.map(lambda i: verify_instance(settings, repos, run_dir, i,
                                                      retry_infra=retry_infra, recheck=recheck), instances):
            for row in rows:
                counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return counts
