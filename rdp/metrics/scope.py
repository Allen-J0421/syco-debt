"""The fixed set of files measured for one instance, and snapshots of that set.

Scope = the files of the instance's human refactoring + every .java file the agent added,
changed or deleted in any turn. The same set is measured at every iteration and for the
human version, so numbers are comparable within an instance.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from rdp.instances import Instance
from rdp.repos import RepoCache
from rdp.util import read_jsonl

HUMAN = "human"


def build_scope(inst: Instance, turn_records: list[dict]) -> list[str]:
    files = {p for p in inst.human_files if p.endswith(".java")}
    for rec in turn_records:
        for path in rec.get("changed_paths") or []:
            if path.endswith(".java"):
                files.add(path)
    return sorted(files)


def version_commits(baseline_sha: str, turn_records: list[dict]) -> dict[int, str]:
    """iteration -> commit. 0 is the untouched baseline; k is the harness commit of turn k."""
    out = {0: baseline_sha}
    for rec in turn_records:
        out[int(rec["turn"])] = rec["commit_sha"]
    return out


def load_turn_records(turns_jsonl: Path) -> list[dict]:
    return sorted(read_jsonl(turns_jsonl), key=lambda r: int(r["turn"]))


def scope_contents(
    repos: RepoCache, inst: Instance, scope: list[str], *, commit: str | None, baseline_sha: str
) -> dict[str, bytes]:
    """Scope file contents for one version. `commit=None` builds the human version:
    the baseline with this instance's human-touched files taken from the human commit."""
    out: dict[str, bytes] = {}
    human = set(inst.human_files)
    for path in scope:
        if commit is None:
            source = inst.commit_id if path in human else baseline_sha
        else:
            source = commit
        data = repos.show_file(inst.project, source, path)
        if data is not None:
            out[path] = data
    return out


def scope_hash(contents: dict[str, bytes]) -> str:
    h = hashlib.sha1()
    for path in sorted(contents):
        h.update(path.encode("utf-8") + b"\0" + hashlib.sha1(contents[path]).digest())
    return h.hexdigest()


def materialise(contents: dict[str, bytes], dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for path, data in contents.items():
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
