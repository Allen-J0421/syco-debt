"""Measure every version (iterations 0..N and the human version) of every finished instance.

Results are cached by a hash of the scoped file contents, so identical snapshots (unchanged
turns, shared baselines, re-runs) are scanned once. The cache is also the resume mechanism.
"""

from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from rdp.config import Settings
from rdp.instances import Instance
from rdp.metrics import aggregate
from rdp.metrics.ck import run_ck
from rdp.metrics.scope import (
    HUMAN,
    build_scope,
    load_turn_records,
    materialise,
    scope_contents,
    scope_hash,
    version_commits,
)
from rdp.metrics.sonar import SonarServer, write_measures
from rdp.repos import RepoCache
from rdp.state import DONE, instance_dir, load_state
from rdp.util import atomic_write_json, eprint, read_json, sanitize_slug

CACHE_VERSION = "v1"


@dataclass(frozen=True)
class Snapshot:
    instance_id: str
    version: str               # "0".."N" or "human"
    commit: str                # "" for the human version
    tree_sha: str
    scope_hash: str
    n_files: int


class Measurer:
    def __init__(self, settings: Settings, repos: RepoCache, server: SonarServer) -> None:
        self.settings = settings
        self.repos = repos
        self.server = server
        cfg = settings.measure
        self.cache_dir = settings.home / "cache" / f"metrics-{CACHE_VERSION}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ck_jar = settings.vendor_dir / "ck.jar"
        self.java = Path(cfg["jdk17_home"]).expanduser() / "bin" / "java"
        self.ck_heap = cfg.get("ck_heap", "-Xmx4g")
        self.timeout = int(cfg.get("scan_timeout_s", 1800))
        self._seq = 0
        self._seq_lock = threading.Lock()

    def cache_path(self, h: str) -> Path:
        return self.cache_dir / f"{h}.json"

    def issues_path(self, h: str) -> Path:
        return self.cache_dir / f"{h}.issues.json"

    def is_cached(self, h: str) -> bool:
        return self.cache_path(h).is_file() and self.issues_path(h).is_file()

    def measure_contents(self, contents: dict[str, bytes], h: str) -> bool:
        """Scan one snapshot into the cache. Returns True on success."""
        if self.is_cached(h):
            return True
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        work = self.settings.scratch_dir / f"m_{h[:12]}_{seq}"
        src = work / "src"
        try:
            materialise(contents, src)
            if not any(p.endswith(".java") for p in contents):
                atomic_write_json(self.cache_path(h), {})
                atomic_write_json(self.issues_path(h), [])
                return True
            ck_out = work / "ck"
            run_ck(self.java, self.ck_jar, src, ck_out, heap=self.ck_heap, timeout_s=self.timeout)
            measures, issues = self.server.scan(f"m_{h[:16]}_{seq}", src)
            sonar_json = work / "sonar.json"
            write_measures(sonar_json, measures)
            atomic_write_json(self.cache_path(h), aggregate.compute(str(ck_out), str(sonar_json)))
            atomic_write_json(self.issues_path(h), issues)
            return True
        except Exception as exc:
            eprint(f"[measure] {h[:12]} failed: {exc}")
            return False
        finally:
            shutil.rmtree(work, ignore_errors=True)


def measure_run(
    settings: Settings, repos: RepoCache, run_dir: Path, instances: list[Instance],
    server: SonarServer, *, jobs: int,
) -> dict[str, int]:
    """Measure all finished instances. Writes scope.json and snapshots.jsonl per instance."""
    measurer = Measurer(settings, repos, server)
    plan: dict[str, tuple[Instance, str | None, list[str], str]] = {}   # hash -> how to rebuild it
    n_snapshots = 0

    for inst in instances:
        state = load_state(run_dir, inst.instance_id)
        if state.status != DONE:
            continue
        idir = instance_dir(run_dir, inst.instance_id)
        model_slug = sanitize_slug(str(settings.agent.get("model")))
        turns = load_turn_records(idir / "logs" / f"{state.stamp}-{model_slug}-turns.jsonl")
        if not turns:
            continue
        scope = build_scope(inst, turns)
        atomic_write_json(idir / "scope.json", {"instance_id": inst.instance_id, "files": scope})
        versions: list[tuple[str, str | None]] = [
            (str(k), c) for k, c in sorted(version_commits(state.baseline_sha, turns).items())
        ] + [(HUMAN, None)]
        snaps: list[dict] = []
        for label, commit in versions:
            contents = scope_contents(repos, inst, scope, commit=commit, baseline_sha=state.baseline_sha)
            h = scope_hash(contents)
            plan.setdefault(h, (inst, commit, scope, state.baseline_sha))
            snaps.append({
                "instance_id": inst.instance_id, "version": label, "commit": commit or "",
                "tree_sha": repos.tree_sha(inst.project, commit) if commit else "",
                "scope_hash": h, "n_files": len(contents),
            })
        (idir / "snapshots.jsonl").write_text("".join(json.dumps(s) + "\n" for s in snaps))
        n_snapshots += len(snaps)

    todo = [h for h in plan if not measurer.is_cached(h)]
    eprint(f"[measure] {n_snapshots} snapshot(s), {len(plan)} unique, {len(todo)} to scan, jobs={jobs}")

    def work(h: str) -> bool:
        inst, commit, scope, baseline = plan[h]
        contents = scope_contents(repos, inst, scope, commit=commit, baseline_sha=baseline)
        return measurer.measure_contents(contents, h)

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(work, h) for h in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            else:
                failed += 1
            if i % 25 == 0:
                eprint(f"[measure] {i}/{len(todo)} scanned ({failed} failed)")
    return {"snapshots": n_snapshots, "unique": len(plan), "scanned": ok, "failed": failed,
            "cached": len(plan) - len(todo)}


def load_cached(settings: Settings, h: str) -> tuple[dict, list]:
    cache_dir = settings.home / "cache" / f"metrics-{CACHE_VERSION}"
    return (read_json(cache_dir / f"{h}.json", {}) or {}, read_json(cache_dir / f"{h}.issues.json", []) or [])
