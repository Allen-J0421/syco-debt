"""SWE-Refactor instances: load, filter, sample, and freeze a selection as a run manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from rdp.util import atomic_write_text, sanitize_slug

# Upstream repositories of the 18 SWE-Refactor projects.
REPO_URLS: dict[str, str] = {
    "checkstyle": "https://github.com/checkstyle/checkstyle.git",
    "commons-lang": "https://github.com/apache/commons-lang.git",
    "commons-io": "https://github.com/apache/commons-io.git",
    "hibernate-orm": "https://github.com/hibernate/hibernate-orm.git",
    "hibernate-search": "https://github.com/hibernate/hibernate-search.git",
    "javaparser": "https://github.com/javaparser/javaparser.git",
    "junit4": "https://github.com/junit-team/junit4.git",
    "junit5": "https://github.com/junit-team/junit5.git",
    "mockito": "https://github.com/mockito/mockito.git",
    "pmd": "https://github.com/pmd/pmd.git",
    "gson": "https://github.com/google/gson.git",
    "guava": "https://github.com/google/guava.git",
    "jadx": "https://github.com/skylot/jadx.git",
    "zxing": "https://github.com/zxing/zxing.git",
    "shiro": "https://github.com/apache/shiro.git",
    "shenyu": "https://github.com/apache/shenyu.git",
    "shardingsphere-elasticjob": "https://github.com/apache/shardingsphere-elasticjob.git",
    "hertzbeat": "https://github.com/apache/hertzbeat.git",
}


@dataclass(frozen=True)
class Instance:
    instance_id: str
    unique_id: str
    project: str
    repo_url: str
    commit_id: str            # the human refactoring commit; the agent starts from its parent
    refactoring_type: str
    target_file: str          # file holding the target method before the refactoring
    file_after: str           # file holding the refactored method after the human commit
    method: str               # simple method name
    method_qualified: str     # pkg.Class#method
    human_files: tuple[str, ...]   # files this instance's human refactoring touches
    compile_jdk: str
    compile_command: str
    is_test_code: bool

    def to_json(self) -> dict:
        d = asdict(self)
        d["human_files"] = list(self.human_files)
        return d

    @classmethod
    def from_json(cls, d: dict) -> Instance:
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        d["human_files"] = tuple(d.get("human_files", ()))
        return cls(**d)


def is_test_path(path: str) -> bool:
    p = "/" + path.lower()
    return "/test/" in p or "/tests/" in p


def _jdk_label(value: object) -> str:
    try:
        f = float(value)  # 11.0, 17.0, 21.0, 1.8
    except (TypeError, ValueError):
        return str(value)
    if abs(f - 1.8) < 1e-6:
        return "8"
    return str(int(f))


def make_instance_id(project: str, commit_id: str, unique_id: str) -> str:
    digest = hashlib.sha1(unique_id.encode("utf-8")).hexdigest()[:6]
    return sanitize_slug(f"{project}-{commit_id[:8]}-{digest}")


def record_to_instance(rec: dict) -> Instance:
    project = rec["projectName"]
    qualified = rec.get("methodNameBefore") or ""
    method = qualified.split("#")[-1] if qualified else ""
    target = rec["filePathBefore"]
    after = rec.get("filePathAfter") or target
    human_files = {target, after}
    for loc in rec.get("diffLocations") or []:
        if isinstance(loc, dict) and loc.get("filePath"):
            human_files.add(loc["filePath"])
    return Instance(
        instance_id=make_instance_id(project, rec["commitId"], rec["uniqueId"]),
        unique_id=rec["uniqueId"],
        project=project,
        repo_url=REPO_URLS.get(project, ""),
        commit_id=rec["commitId"],
        refactoring_type=rec["type"],
        target_file=target,
        file_after=after,
        method=method,
        method_qualified=qualified,
        human_files=tuple(sorted(human_files)),
        compile_jdk=_jdk_label(rec.get("compileJDK")),
        compile_command=rec.get("compileCommand") or "",
        is_test_code=is_test_path(target),
    )


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_instances(
    records: list[dict],
    *,
    include_test_code: bool = False,
    require_compile: bool = True,
    one_per_commit_file: bool = True,
    types: set[str] | None = None,
    projects: set[str] | None = None,
    ids: set[str] | None = None,
) -> list[Instance]:
    """Apply the default filters. Order is deterministic (sorted by instance id)."""
    out: list[Instance] = []
    seen_commit_file: set[tuple[str, str]] = set()
    for rec in sorted(records, key=lambda r: r["uniqueId"]):
        if require_compile and not (
            rec.get("compileResultBefore") is True and rec.get("compileResultCurrent") is True
        ):
            continue
        inst = record_to_instance(rec)
        if not inst.repo_url or not inst.method:
            continue
        if not include_test_code and inst.is_test_code:
            continue
        if types and inst.refactoring_type not in types:
            continue
        if projects and inst.project not in projects:
            continue
        if ids and inst.instance_id not in ids and inst.unique_id not in ids:
            continue
        key = (inst.commit_id, inst.target_file)
        if one_per_commit_file:
            if key in seen_commit_file:
                continue
            seen_commit_file.add(key)
        out.append(inst)
    return sorted(out, key=lambda i: i.instance_id)


def stratified_sample(
    instances: list[Instance], n: int, *, stratify: tuple[str, ...], seed: int
) -> list[Instance]:
    """Deterministic sample of n: shuffle within strata, then draw round-robin across strata."""
    if n >= len(instances):
        return list(instances)
    rng = random.Random(seed)
    strata: dict[tuple, list[Instance]] = defaultdict(list)
    for inst in instances:
        strata[tuple(getattr(inst, k) for k in stratify)].append(inst)
    keys = sorted(strata)
    for key in keys:
        rng.shuffle(strata[key])
    rng.shuffle(keys)
    picked: list[Instance] = []
    while len(picked) < n:
        for key in keys:
            if strata[key] and len(picked) < n:
                picked.append(strata[key].pop())
    return sorted(picked, key=lambda i: i.instance_id)


def write_manifest(path: Path, instances: list[Instance]) -> None:
    text = "".join(json.dumps(i.to_json(), ensure_ascii=False) + "\n" for i in instances)
    atomic_write_text(path, text)


def read_manifest(path: Path) -> list[Instance]:
    out: list[Instance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Instance.from_json(json.loads(line)))
    return out


INSTANCE_COLUMNS = [
    "instance_id", "unique_id", "project", "commit_id", "refactoring_type", "target_file",
    "file_after", "method", "method_qualified", "compile_jdk", "is_test_code",
]


def write_instances_csv(path: Path, instances: list[Instance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INSTANCE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for inst in instances:
            w.writerow(inst.to_json())
