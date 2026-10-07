"""End-to-end runs against a fake Claude CLI: parallel scheduling, limit pause, resume, tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import TARGET, make_instance

from rdp.assemble import assemble, label_sample
from rdp.config import load_settings
from rdp.runner.claude_agent import ClaudeAgent
from rdp.runner.experiment import RunContext
from rdp.runner.limit_detect import LimitDetectConfig
from rdp.runner.models import ClarificationPatterns
from rdp.runner.prompt_source import FixedPromptSource
from rdp.scheduler import PauseConfig, run_all
from rdp.state import load_state, save_state
from rdp.util import read_jsonl

import re

PATTERNS = ClarificationPatterns(
    refusal_blockers=(re.compile(r"\bi'm not (going to|refactoring)\b", re.I),),
    clarification_signals=(re.compile(r"\bcould you\b", re.I),),
)
FAST = PauseConfig(buffer_minutes=0, fixed_wait_minutes=0.01, max_wait_minutes=1, poll_seconds=0.1)


def make_ctx(tmp_path: Path, repos, turns: int) -> RunContext:
    return RunContext(
        run_dir=tmp_path / "home" / "runs" / "t", repos=repos, model="claude-opus-4-8", turns=turns,
        turn1_template="Refactor the method `{method}` in `{file}`.", prompter_label="fixed",
        agent_factory=lambda wd: ClaudeAgent(wd, "claude-opus-4-8", timeout=60, effort="high"),
        prompt_source_factory=lambda first: FixedPromptSource(first, "refactor"),
        clarification_patterns=PATTERNS, limit_cfg=LimitDetectConfig(),
    )


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_parallel_run_with_limit_pause(tmp_path, upstream, repos, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_PLAN", "edit,limit,edit,edit")
    repos.ensure_mirror("demo", upstream["url"], commits=[upstream["human"]])
    instances = [make_instance(upstream, n) for n in range(6)]
    ctx = make_ctx(tmp_path, repos, turns=4)

    counts = run_all(ctx, instances, jobs=3, pause=FAST, max_requeues=3)
    assert counts == {"done": 6}

    events = read_jsonl(ctx.run_dir / "events.jsonl")
    limited = [e for e in events if e["event"] == "provider_limit"]
    assert len(limited) == 1                                    # the limit fired once, one run discarded
    branches = set()
    for inst in instances:
        st = load_state(ctx.run_dir, inst.instance_id)
        assert st.status == "done" and st.turns_done == 4
        idir = ctx.run_dir / "instances" / inst.instance_id
        logs = list((idir / "logs").glob("*-log.csv"))
        assert len(logs) == 1                                   # the discarded attempt left no log
        log = rows(logs[0])
        assert [r["run"] for r in log] == ["1", "2", "3", "4"]
        assert all(int(r["lines_total"]) == 1 for r in log)
        run_dirs = sorted(p.name for p in (idir / logs[0].name.replace("-log.csv", "")).iterdir())
        assert run_dirs == ["run_001", "run_002", "run_003", "run_004"]
        assert repos.branch_exists("demo", st.branch)
        assert not repos.worktree_path(inst.instance_id).exists()
        branches.add(st.branch)
        # every commit on the branch touches only this instance's edits
        text = repos.show_file("demo", st.branch, TARGET).decode()
        assert text.count("// edit") == 4
        for d in st.discarded:
            assert not repos.branch_exists("demo", d["branch"])
    assert len(branches) == 6
    discarded = [d for inst in instances for d in load_state(ctx.run_dir, inst.instance_id).discarded]
    assert len(discarded) == 1 and discarded[0]["reason"] == "provider_limit"

    # resume: nothing left to do
    assert run_all(ctx, instances, jobs=3, pause=FAST) == {"done": 6}
    assert len(read_jsonl(ctx.run_dir / "events.jsonl")) == len(events)


def test_stale_running_attempt_is_discarded_and_rerun(tmp_path, upstream, repos, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_PLAN", "edit")
    repos.ensure_mirror("demo", upstream["url"], commits=[upstream["human"]])
    inst = make_instance(upstream, 1)
    ctx = make_ctx(tmp_path, repos, turns=2)
    assert run_all(ctx, [inst], jobs=1, pause=FAST) == {"done": 1}
    st = load_state(ctx.run_dir, inst.instance_id)
    old_branch = st.branch
    st.status = "running"                                       # as if the process had been killed
    save_state(ctx.run_dir, st)
    assert run_all(ctx, [inst], jobs=1, pause=FAST) == {"done": 1}
    st2 = load_state(ctx.run_dir, inst.instance_id)
    assert st2.branch != old_branch or st2.attempts == 2
    assert len(list((ctx.run_dir / "instances" / inst.instance_id / "logs").glob("*-log.csv"))) == 1


def test_signals_tables_and_self_commit(tmp_path, upstream, repos, fake_claude, monkeypatch):
    # turn: 1 edit, 2 stop (declines), 3 declines but edits, 4 edits and commits by itself,
    #       5 asks a question (zero change) then edits after the reply, 6 fails
    monkeypatch.setenv("FAKE_CLAUDE_PLAN", "edit,stop,claimedit,selfcommit,ask,fail")
    repos.ensure_mirror("demo", upstream["url"], commits=[upstream["human"]])
    inst = make_instance(upstream, 1)
    ctx = make_ctx(tmp_path, repos, turns=6)
    assert run_all(ctx, [inst], jobs=1, pause=FAST) == {"done": 1}

    settings = load_settings(home=str(tmp_path / "home"))
    counts = assemble(settings, ctx.run_dir, [inst])
    assert counts["steps"] == 6 and counts["signals"] == 1
    tables = ctx.run_dir / "tables"
    steps = {int(r["turn"]): r for r in rows(tables / "steps.csv")}
    assert [int(steps[t]["lines_total"]) for t in range(1, 7)] == [1, 0, 1, 1, 1, 0]
    assert steps[4]["lines_total"] == "1"                       # the agent's own commit still counts
    assert steps[1]["target_lines_total"] == "1" and steps[1]["java_lines_total"] == "1"
    assert steps[2]["stop"] == "1" and steps[2]["stop_claim"] == "1" and steps[2]["refusal_edit"] == "0"
    assert steps[3]["refusal_edit"] == "1" and steps[3]["post_stop_edit"] == "1"
    assert steps[5]["clarification_followup"] == "1"
    assert steps[6]["exit_code"] == "7"
    assert steps[1]["prompt_source"] == "template" and steps[2]["prompt_source"] == "fixed"
    assert float(steps[5]["usage_cost_usd"]) == 0.02            # two agent calls in the follow-up turn
    assert "`sum`" in read_jsonl(next((ctx.run_dir / "instances" / inst.instance_id / "logs")
                                      .glob("*-turns.jsonl")))[0]["prompt"]

    sig = rows(tables / "signals.csv")[0]
    assert (sig["t_star"], sig["never_stopped"], sig["relapsed"]) == ("2", "0", "1")
    # Turn 5 counts as a stop claim that still edits: the question is stop-claim language
    # (the "questioning" category) and the turn ends with an edit. Same rule as the source.
    assert sig["post_stop_lines"] == "3" and sig["refusal_edit_turns"] == "2"
    assert sig["stop_claim_turns"] == "3" and abs(float(sig["s3_refusal_edit_rate"]) - 2 / 3) < 1e-9
    assert steps[5]["claim_categories"] == "QUESTIONING"
    assert sig["failed_turns"] == "1" and sig["clean"] == "0"

    assert label_sample(settings, ctx.run_dir, [inst], n=4, seed=0) == 4
    sheet = rows(tables / "label_sample.csv")
    assert {r["detector_stop_claim"] for r in sheet} == {"0", "1"}

    # iteration commits come from the log, even though the agent added a commit of its own
    st = load_state(ctx.run_dir, inst.instance_id)
    turns = read_jsonl(next((ctx.run_dir / "instances" / inst.instance_id / "logs").glob("*-turns.jsonl")))
    assert turns[3]["parent_sha"] == turns[2]["commit_sha"]
    assert repos.rev_parse("demo", st.branch) == turns[-1]["commit_sha"]


class StubServer:
    """Stands in for SonarQube: reports ncloc = lines of Java and one 5-minute smell per file."""

    def scan(self, key, src_dir):
        files = sorted(Path(src_dir).rglob("*.java"))
        ncloc = sum(len(f.read_text().splitlines()) for f in files)
        measures = {"component": {"measures": [
            {"metric": "ncloc", "value": str(ncloc)}, {"metric": "files", "value": str(len(files))},
            {"metric": "functions", "value": "2"}, {"metric": "complexity", "value": "4"},
            {"metric": "cognitive_complexity", "value": "2"}, {"metric": "code_smells", "value": str(len(files))},
            {"metric": "duplicated_lines_density", "value": "0.0"}, {"metric": "sqale_rating", "value": "1.0"},
            {"metric": "sqale_index", "value": str(5 * len(files))}, {"metric": "sqale_debt_ratio", "value": "1.0"},
        ]}}
        issues = [{"rule": "java:S100", "severity": "MINOR", "effort_min": 5,
                   "file": str(f.relative_to(src_dir)), "line": 1, "message": "m"} for f in files]
        return measures, issues


def test_measure_assemble_and_verify(tmp_path, upstream, repos, fake_claude, monkeypatch):
    from rdp.metrics import measure as measure_mod
    from rdp.metrics.aggregate import METRIC_COLS
    from rdp.verify import verify_run

    monkeypatch.setenv("FAKE_CLAUDE_PLAN", "edit,stop,edit")
    repos.ensure_mirror("demo", upstream["url"], commits=[upstream["human"]])
    instances = [make_instance(upstream, n) for n in range(2)]
    ctx = make_ctx(tmp_path, repos, turns=3)
    assert run_all(ctx, instances, jobs=2, pause=FAST) == {"done": 2}

    home = tmp_path / "home"
    jdk = tmp_path / "jdk17"
    (jdk / "bin").mkdir(parents=True)
    (jdk / "bin" / "java").write_text("#!/bin/sh\n")
    cfg = tmp_path / "run.toml"
    base = (Path(__file__).resolve().parent.parent / "config" / "run.toml").read_text()
    cfg.write_text(base.replace('"17" = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"',
                                f'"17" = "{jdk}"'))
    settings = load_settings(cfg, home=str(home))
    object.__setattr__(settings, "config_dir", Path(__file__).resolve().parent.parent / "config")

    def fake_ck(java, jar, src, out, **kw):
        out.mkdir(parents=True, exist_ok=True)
        (out / "class.csv").write_text("file,class,cbo\nx,demo.Calc,3\n")
        (out / "method.csv").write_text("file,class,method,wmc\nx,demo.Calc,sum,2\n")
        return True

    monkeypatch.setattr(measure_mod, "run_ck", fake_ck)
    result = measure_mod.measure_run(settings, repos, ctx.run_dir, instances, StubServer(), jobs=2)
    assert result["failed"] == 0 and result["snapshots"] == 2 * 5       # iterations 0-3 + human, each
    again = measure_mod.measure_run(settings, repos, ctx.run_dir, instances, StubServer(), jobs=2)
    assert again["scanned"] == 0 and again["cached"] == again["unique"]  # everything from the cache

    counts = assemble(settings, ctx.run_dir, instances)
    assert counts["metrics"] == 10
    metrics = rows(ctx.run_dir / "tables" / "metrics.csv")
    foundational = ["unit", "unit_type", "tool", "model", "timestamp", "branch", "iteration", "commit_sha",
                    "tree_sha"] + METRIC_COLS
    assert list(metrics[0])[: len(foundational)] == foundational       # same columns as data/metrics.csv
    one = [r for r in metrics if r["unit"] == instances[0].instance_id]
    agent = {r["iteration"]: r for r in one if r["arm"] == "agent"}
    human = [r for r in one if r["arm"] == "human"]
    assert sorted(agent) == ["0", "1", "2", "3"] and len(human) == 1
    assert agent["1"]["tree_sha"] == agent["2"]["tree_sha"]             # the stop turn left the tree alone
    assert agent["1"]["scope_hash"] == agent["2"]["scope_hash"] != agent["3"]["scope_hash"]
    assert int(agent["3"]["ncloc"]) == int(agent["0"]["ncloc"]) + 2
    assert int(human[0]["ncloc"]) > int(agent["0"]["ncloc"]) and human[0]["commit_sha"] == upstream["human"]
    assert agent["0"]["cbo_mean"] == "3.0" and agent["0"]["tech_debt_min"] == "5"
    issues = rows(ctx.run_dir / "tables" / "issues.csv")
    assert len(issues) == 10 and {i["rule"] for i in issues} == {"java:S100"}
    sig = rows(ctx.run_dir / "tables" / "signals.csv")[0]
    assert float(sig["s2_post_stop"]) == 1 / float(agent["0"]["ncloc"])  # post-stop lines / baseline size

    vcounts = verify_run(settings, repos, ctx.run_dir, instances, jobs=2)
    assert vcounts == {"pass": 2 * 4}                                    # versions 0, 1, 2 (first stop), 3
    assert verify_run(settings, repos, ctx.run_dir, instances, jobs=2) == {}   # resumes: nothing left
    assert assemble(settings, ctx.run_dir, instances)["verify"] == 8
