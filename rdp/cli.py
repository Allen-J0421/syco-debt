"""Command line: python -m rdp <command> --run <name> ..."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdp import doctor
from rdp.config import Settings, load_settings
from rdp.instances import (
    Instance,
    load_dataset,
    read_manifest,
    select_instances,
    stratified_sample,
    write_instances_csv,
    write_manifest,
)
from rdp.repos import RepoCache
from rdp.state import load_state
from rdp.util import eprint, sanitize_slug


# ---- shared helpers -------------------------------------------------------------
def _settings(args) -> Settings:
    return load_settings(Path(args.config) if args.config else None, home=args.home)


def _run_dir(settings: Settings, args) -> Path:
    return settings.run_dir(args.run)


def _repos(settings: Settings) -> RepoCache:
    return RepoCache(settings.repos_dir, settings.worktrees_dir)


def _manifest(settings: Settings, args) -> list[Instance]:
    path = _run_dir(settings, args) / "manifest.jsonl"
    if not path.is_file():
        raise SystemExit(f"no manifest for run {args.run!r}: run `python -m rdp prepare --run {args.run}` first")
    instances = read_manifest(path)
    ids = set(args.ids.split(",")) if getattr(args, "ids", None) else None
    if ids:
        instances = [i for i in instances if i.instance_id in ids]
    limit = getattr(args, "limit", None)
    if limit:
        instances = instances[:limit]
    return instances


def _ensure_mirrors(repos: RepoCache, instances: list[Instance]) -> None:
    by_project: dict[str, tuple[str, list[str]]] = {}
    for inst in instances:
        by_project.setdefault(inst.project, (inst.repo_url, []))[1].append(inst.commit_id)

    def one(item):
        project, (url, commits) = item
        if not (repos.mirror_path(project) / "HEAD").is_file():
            eprint(f"[clone] {project} <- {url}")
        repos.ensure_mirror(project, url, commits=sorted(set(commits)))
        missing = [c for c in set(commits) if not repos.has_commit(project, c)]
        if missing:
            eprint(f"[clone] {project}: {len(missing)} commit(s) not found upstream, e.g. {missing[0]}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, by_project.items()))


def _load_patterns(path: Path):
    patterns = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.I | re.M))
    return tuple(patterns)


# ---- commands -------------------------------------------------------------------
def cmd_doctor(args) -> int:
    return doctor.report(_settings(args))


def cmd_prepare(args) -> int:
    settings = _settings(args)
    run_dir = _run_dir(settings, args)
    manifest = run_dir / "manifest.jsonl"
    if manifest.is_file() and not args.force:
        raise SystemExit(f"{manifest} exists; the selection is frozen. Use --force to replace it.")
    ids = None
    if args.ids_file:
        ids = {line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()}
    pool = select_instances(
        load_dataset(settings.dataset),
        include_test_code=args.include_test_code,
        one_per_commit_file=not args.allow_same_file,
        types=set(args.types.split(",")) if args.types else None,
        projects=set(args.projects.split(",")) if args.projects else None,
        ids=ids,
    )
    chosen = pool if (args.all or args.sample is None) else stratified_sample(
        pool, args.sample, stratify=tuple(args.stratify.split(",")), seed=args.seed)
    write_manifest(manifest, chosen)
    write_instances_csv(run_dir / "tables" / "instances.csv", chosen)
    by_type: dict[str, int] = {}
    for inst in chosen:
        by_type[inst.refactoring_type] = by_type.get(inst.refactoring_type, 0) + 1
    print(f"eligible: {len(pool)}   selected: {len(chosen)}   projects: {len({i.project for i in chosen})}")
    for name, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {name}")
    print(f"manifest: {manifest}")
    return 0


def cmd_clone(args) -> int:
    settings = _settings(args)
    _ensure_mirrors(_repos(settings), _manifest(settings, args))
    return 0


def _run_context(settings: Settings, args, run_dir: Path, repos: RepoCache):
    from rdp.runner import prompter as prompter_mod
    from rdp.runner.claude_agent import ClaudeAgent
    from rdp.runner.experiment import RunContext
    from rdp.runner.limit_detect import LimitDetectConfig
    from rdp.runner.models import ClarificationPatterns, PrompterConfig
    from rdp.runner.prompt_source import FixedPromptSource, GeminiPromptSource

    agent_cfg, prompter_cfg = settings.agent, settings.prompter
    model = str(agent_cfg.get("model"))
    effort = agent_cfg.get("effort") or None
    timeout = int(agent_cfg.get("turn_timeout_s", 2700))
    cdir = settings.config_dir
    template = (cdir / "turn1_prompt.txt").read_text(encoding="utf-8")
    patterns = ClarificationPatterns(
        refusal_blockers=_load_patterns(cdir / "clarification_refusal_patterns.txt"),
        clarification_signals=_load_patterns(cdir / "clarification_signal_patterns.txt"),
        require_question_mark=True,
    )
    fallback = str(prompter_cfg.get("fallback_prompt", "refactor"))
    if args.fixed_prompt:
        label = "fixed"

        def source_factory(first_request: str):
            return FixedPromptSource(first_request, fallback)
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is not set (put it in .env). The simulated developer needs it; "
                             "--fixed-prompt runs without it but is not the experimental condition.")
        pconf = PrompterConfig(
            api_key=api_key, model=str(prompter_cfg.get("model")),
            system_prompt=(cdir / "prompter_system_prompt.txt").read_text(encoding="utf-8").strip(),
            nudge=str(prompter_cfg.get("nudge")), fallback_prompt=fallback,
            clarification_nudge=str(prompter_cfg.get("clarification_nudge")),
            retries=int(prompter_cfg.get("retries", 3)),
        )
        prompter_mod.set_max_concurrent(int(prompter_cfg.get("max_concurrent", 4)))
        label = sanitize_slug(pconf.model)

        def source_factory(first_request: str):
            return GeminiPromptSource(first_request, pconf)

    return RunContext(
        run_dir=run_dir, repos=repos, model=model,
        turns=args.turns or int(agent_cfg.get("turns", 10)),
        turn1_template=template, prompter_label=label,
        agent_factory=lambda work_dir: ClaudeAgent(work_dir, model, timeout=timeout, effort=effort),
        prompt_source_factory=source_factory,
        clarification_patterns=patterns, limit_cfg=LimitDetectConfig(),
        keep_worktree=args.keep_worktrees,
    )


def cmd_run(args) -> int:
    from rdp.scheduler import PauseConfig, run_all

    settings = _settings(args)
    run_dir, repos = _run_dir(settings, args), _repos(settings)
    instances = _manifest(settings, args)
    ctx = _run_context(settings, args, run_dir, repos)
    _ensure_mirrors(repos, instances)
    runnable = [i for i in instances if repos.has_commit(i.project, i.commit_id)]
    if len(runnable) < len(instances):
        eprint(f"[run] skipping {len(instances) - len(runnable)} instance(s) whose commit is unavailable")
    rcfg = settings.run
    pause = PauseConfig(
        buffer_minutes=float(rcfg.get("limit_buffer_minutes", 2)),
        fixed_wait_minutes=float(rcfg.get("limit_fixed_wait_minutes", 60)),
        max_wait_minutes=float(rcfg.get("limit_max_wait_minutes", 480)),
    )
    counts = run_all(ctx, runnable, jobs=args.jobs or int(rcfg.get("jobs", 4)), pause=pause,
                     max_requeues=int(rcfg.get("max_requeues", 3)), retry_failed=args.retry_failed)
    print("run finished: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0 if set(counts) <= {"done"} else 1


def cmd_status(args) -> int:
    settings = _settings(args)
    run_dir = _run_dir(settings, args)
    instances = _manifest(settings, args)
    counts: dict[str, int] = {}
    turns = fallbacks = requeues = 0
    for inst in instances:
        st = load_state(run_dir, inst.instance_id)
        counts[st.status] = counts.get(st.status, 0) + 1
        turns += st.turns_done
        fallbacks += st.prompter_fallbacks
        requeues += st.requeues
        if args.verbose or st.status == "failed":
            print(f"  {st.status:8s} {inst.instance_id}  turns={st.turns_done}"
                  + (f"  error={st.error}" if st.error else ""))
    print(f"run {args.run}: {len(instances)} instance(s)  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"turns completed: {turns}   provider-limit discards: {requeues}   prompter fallbacks: {fallbacks}")
    return 0


def cmd_setup_metrics(args) -> int:
    from rdp.metrics.setup import setup_vendor

    settings = _settings(args)
    setup_vendor(settings.vendor_dir, str(settings.measure.get("sonar_version")))
    return 0


def _sonar_server(settings: Settings):
    from rdp.metrics.sonar import SonarServer

    cfg = settings.measure
    return SonarServer(
        host=str(cfg.get("sonar_host", "http://localhost:9000")),
        sonar_home=settings.vendor_dir / "sonarqube",
        jdk17_home=Path(str(cfg.get("jdk17_home"))).expanduser(),
        token_file=settings.vendor_dir / ".sonar.token",
        dummy_bin=settings.vendor_dir / "dummybin",
        work_dir=settings.scratch_dir / "sonar",
        scanner_heap=str(cfg.get("scanner_heap", "-Xmx2g")),
        scan_timeout_s=int(cfg.get("scan_timeout_s", 1800)),
    )


def cmd_measure(args) -> int:
    from rdp.metrics.measure import measure_run

    settings = _settings(args)
    server = _sonar_server(settings)
    server.start()
    server.ensure_token()
    result = measure_run(settings, _repos(settings), _run_dir(settings, args), _manifest(settings, args),
                         server, jobs=args.jobs or int(settings.measure.get("jobs", 3)))
    print("measure: " + ", ".join(f"{k}={v}" for k, v in result.items()))
    if args.stop_server:
        server.stop()
    return 0 if result["failed"] == 0 else 1


def cmd_verify(args) -> int:
    from rdp.verify import verify_run

    settings = _settings(args)
    counts = verify_run(settings, _repos(settings), _run_dir(settings, args), _manifest(settings, args),
                        jobs=args.jobs or int(settings.verify.get("jobs", 2)))
    print("verify: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to do"))
    return 0


def cmd_assemble(args) -> int:
    from rdp.assemble import assemble

    settings = _settings(args)
    run_dir = _run_dir(settings, args)
    counts = assemble(settings, run_dir, _manifest(settings, args))
    print(f"tables in {run_dir / 'tables'}: " + ", ".join(f"{k}={v} rows" for k, v in counts.items()))
    return 0


def cmd_label_sample(args) -> int:
    from rdp.assemble import label_sample

    settings = _settings(args)
    run_dir = _run_dir(settings, args)
    n = label_sample(settings, run_dir, _manifest(settings, args), n=args.n, seed=args.seed)
    print(f"wrote {n} turn(s) to {run_dir / 'tables' / 'label_sample.csv'}")
    return 0


def cmd_export(args) -> int:
    settings = _settings(args)
    src = _run_dir(settings, args) / "tables"
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(src.glob("*.csv")):
        target = dest / path.name
        if target.exists() and not args.force:
            raise SystemExit(f"{target} exists; use --force to overwrite")
        shutil.copy2(path, target)
        copied.append(path.name)
    print(f"copied to {dest}: " + ", ".join(copied))
    return 0


def cmd_purge(args) -> int:
    """Delete a run: its artifacts, its worktrees, and its experiment branches in the mirrors."""
    settings = _settings(args)
    run_dir, repos = _run_dir(settings, args), _repos(settings)
    if not args.yes:
        raise SystemExit(f"this deletes {run_dir} and the run's branches; repeat with --yes")
    n = 0
    for inst in _manifest(settings, args):
        st = load_state(run_dir, inst.instance_id)
        if not (repos.mirror_path(inst.project) / "HEAD").is_file():
            continue
        repos.remove_worktree(inst.project, inst.instance_id)
        for branch in [st.branch] + [d.get("branch", "") for d in st.discarded]:
            if branch and repos.branch_exists(inst.project, branch):
                repos.delete_branch(inst.project, branch)
                n += 1
    shutil.rmtree(run_dir, ignore_errors=True)
    print(f"purged run {args.run}: {n} branch(es) deleted")
    return 0


# ---- parser ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m rdp", description="Refactor-debt pipeline on SWE-Refactor")
    p.add_argument("--config", help="path to run.toml (default: config/run.toml)")
    p.add_argument("--home", help="runtime home (default: RDP_HOME or the config value)")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_: str, *, run: bool = True, select: bool = False, jobs: bool = False):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=func)
        if run:
            sp.add_argument("--run", required=True, help="run name (a folder under <home>/runs)")
        if select:
            sp.add_argument("--limit", type=int, help="only the first N instances of the manifest")
            sp.add_argument("--ids", help="comma-separated instance ids")
        if jobs:
            sp.add_argument("--jobs", type=int, help="parallel workers (default from run.toml)")
        return sp

    add("doctor", cmd_doctor, "check tools and settings", run=False)

    sp = add("prepare", cmd_prepare, "select instances and freeze them as the run's manifest")
    sp.add_argument("--sample", type=int, help="stratified sample of N (default: all eligible)")
    sp.add_argument("--all", action="store_true", help="every eligible instance")
    sp.add_argument("--stratify", default="refactoring_type,project")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--types", help="comma-separated refactoring types")
    sp.add_argument("--projects", help="comma-separated project names")
    sp.add_argument("--ids-file", help="file of instance ids or dataset uniqueIds, one per line")
    sp.add_argument("--include-test-code", action="store_true")
    sp.add_argument("--allow-same-file", action="store_true",
                    help="keep several instances from the same commit and file")
    sp.add_argument("--force", action="store_true")

    add("clone", cmd_clone, "clone or update the project mirrors the run needs", select=True)

    sp = add("run", cmd_run, "run the agent on every pending instance, in parallel", select=True, jobs=True)
    sp.add_argument("--turns", type=int)
    sp.add_argument("--retry-failed", action="store_true")
    sp.add_argument("--fixed-prompt", action="store_true",
                    help="no simulated developer: repeat the fallback prompt (dry runs only)")
    sp.add_argument("--keep-worktrees", action="store_true")

    sp = add("status", cmd_status, "progress of a run", select=True)
    sp.add_argument("-v", "--verbose", action="store_true")

    add("setup-metrics", cmd_setup_metrics, "download CK and SonarQube into <home>/vendor", run=False)
    sp = add("measure", cmd_measure, "SonarQube + CK on every version of every finished instance",
             select=True, jobs=True)
    sp.add_argument("--stop-server", action="store_true", help="stop SonarQube afterwards")
    add("verify", cmd_verify, "compile and test selected versions", select=True, jobs=True)
    add("assemble", cmd_assemble, "build metrics/steps/signals/issues/verify tables", select=True)
    add("signals", cmd_assemble, "alias of assemble (signals need no metrics)", select=True)
    sp = add("label-sample", cmd_label_sample, "sheet for hand-checking the stop-claim detector", select=True)
    sp.add_argument("--n", type=int, default=150)
    sp.add_argument("--seed", type=int, default=0)
    sp = add("purge", cmd_purge, "delete a run, its worktrees and its branches")
    sp.add_argument("--yes", action="store_true")
    sp = add("export", cmd_export, "copy the run's tables to another folder")
    sp.add_argument("--dest", required=True)
    sp.add_argument("--force", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
