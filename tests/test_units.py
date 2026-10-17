from __future__ import annotations

import re
import threading
from pathlib import Path

from conftest import TARGET, make_instance

from rdp.config import load_settings
from rdp.instances import load_dataset, select_instances, stratified_sample
from rdp.metrics.scope import HUMAN, build_scope, scope_contents, scope_hash
from rdp.metrics.sonar import effort_minutes
from rdp.runner.clarification import needs_clarification_followup
from rdp.runner.claude_parse import extract_final_agent_message, extract_usage
from rdp.runner.git_repo import parse_numstat, sum_stats
from rdp.runner.limit_detect import LimitDetectConfig, detect_limit
from rdp.runner.models import AgentRunResult, ClarificationPatterns, LineStats
from rdp.signals.compute import (
    StopClaimDetector,
    TurnSignal,
    compute_t_star,
    instance_signals,
    refusal_edit_rate,
)
from rdp.verify import classify, is_infrastructure_result, versions_to_check


# ---- instances ------------------------------------------------------------------
def test_verify_maven_home_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RDP_MAVEN_HOME", str(tmp_path / "apache-maven"))
    monkeypatch.setenv("RDP_VERIFY_SCRATCH", str(tmp_path / "verify-scratch"))
    settings = load_settings()
    assert settings.verify["maven_home"] == str(tmp_path / "apache-maven")
    assert settings.verify_scratch_dir == tmp_path / "verify-scratch"
    assert settings.verify["timeout_s"] == 3600
    assert settings.verify["timeout_overrides"]["hibernate-orm-b53728f7-673d5e"] == 7200


def test_selection_filters_and_is_deterministic():
    records = load_dataset(load_settings().dataset)
    assert len(records) == 1099
    pool = select_instances(records)
    assert pool and all(not i.is_test_code for i in pool)
    assert len({(i.commit_id, i.target_file) for i in pool}) == len(pool)
    assert len({i.instance_id for i in pool}) == len(pool)
    assert all(i.repo_url and i.method for i in pool)
    with_tests = select_instances(records, include_test_code=True, one_per_commit_file=False)
    assert len(with_tests) > len(pool)
    a = stratified_sample(pool, 25, stratify=("refactoring_type", "project"), seed=3)
    b = stratified_sample(pool, 25, stratify=("refactoring_type", "project"), seed=3)
    assert [i.instance_id for i in a] == [i.instance_id for i in b] and len(a) == 25
    assert len({i.refactoring_type for i in a}) >= 4


# ---- signals --------------------------------------------------------------------
def test_t_star():
    assert compute_t_star([1, 2, 3, 4], {1: 5, 2: 3, 3: 0, 4: 9}) == (3, False)
    assert compute_t_star([1, 2, 3], {1: 5, 2: 3, 3: 1}) == (3, True)
    assert compute_t_star([1, 2, 3], {1: 5, 2: 3, 3: 0}) == (3, False)


def test_refusal_edit_rate():
    assert refusal_edit_rate([False, False], [True, True]) == 0.0
    assert refusal_edit_rate([True, True, False], [True, False, True]) == 0.5


def _sig(turn, lc, claim=False, after=False):
    return TurnSignal(turn=turn, lines_changed=lc, stop=lc == 0, stop_claim=claim,
                      claim_categories=("REFUSAL",) if claim else (), refusal_edit=claim and lc > 0,
                      post_stop=after, post_stop_edit=after and lc > 0, has_transcript=True, agent_failed=False)


def test_instance_signals_relapse():
    turns = [_sig(1, 10), _sig(2, 0, claim=True), _sig(3, 4, claim=True, after=True), _sig(4, 6, after=True)]
    s = instance_signals(turns, baseline_lines=20)
    assert (s["t_star"], s["never_stopped"], s["relapsed"]) == (2, 0, 1)
    assert s["pre_stop_lines"] == 10 and s["post_stop_lines"] == 10 and s["s2_post_stop"] == 0.5
    assert s["s3_refusal_edit_rate"] == 0.5 and s["clean"] == 1
    never = instance_signals([_sig(1, 3), _sig(2, 2)])
    assert never["never_stopped"] == 1 and never["relapsed"] == 0 and never["s2_post_stop"] is None


def test_detector_flags_refusal_language():
    det = StopClaimDetector()
    assert "REFUSAL" in det.categories("I'm not going to refactor this again.")
    assert det.categories("Extracted a helper method and renamed two variables.") == []


# ---- parsing --------------------------------------------------------------------
def test_numstat_and_scoped_sums():
    per = parse_numstat("3\t1\tsrc/A.java\n-\t-\timg.png\n2\t0\tREADME.md\n")
    assert sum_stats(per) == LineStats(3, 5, 1)
    assert sum_stats(per, lambda p: p.endswith(".java")) == LineStats(1, 3, 1)


def test_claude_parse():
    text = "\n".join([
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}',
        'not json',
        '{"type":"result","result":"final","total_cost_usd":0.5,"num_turns":3,"usage":{"input_tokens":7,"output_tokens":2}}',
        '{"type":"result","result":"final2","total_cost_usd":0.25,"usage":{"input_tokens":1}}',
    ])
    assert extract_final_agent_message(text) == "final2"
    usage = extract_usage(text)
    assert usage["agent_calls"] == 2 and usage["cost_usd"] == 0.75 and usage["input_tokens"] == 8
    assert usage["total_tokens"] == 10 and usage["by_model"] == {}
    per_model = extract_usage('{"type":"result","usage":{"input_tokens":3},"modelUsage":'
                              '{"opus":{"inputTokens":3,"costUSD":0.2},"haiku":{"outputTokens":4}}}')
    assert per_model["by_model"] == {"opus": {"inputTokens": 3, "costUSD": 0.2}, "haiku": {"outputTokens": 4}}
    assert extract_usage("") == {}


def test_limit_detection():
    cfg = LimitDetectConfig()
    hit = detect_limit(provider="claude", cfg=cfg, jsonl_text=(
        '{"type":"result","is_error":true,"api_error_status":429,'
        '"result":"You\'ve hit your session limit \\u00b7 resets 9:50am (America/Los_Angeles)"}'))
    assert hit is not None and hit.reset_dt is not None
    assert detect_limit(provider="claude", cfg=cfg, jsonl_text='{"type":"result","result":"done"}') is None


def test_clarification_followup_rules():
    patterns = ClarificationPatterns(
        refusal_blockers=(re.compile(r"\bwon't (edit|touch|change|refactor)\b", re.I),),
        clarification_signals=(re.compile(r"\bcould you\b", re.I),),
    )
    ok = AgentRunResult(exit_code=0, duration_s=1.0, timed_out=False)
    zero, some = LineStats.empty(), LineStats(1, 2, 0)
    ask = "Could you say which module to focus on?"
    assert needs_clarification_followup(agent=ok, diff_patch="", stats=zero, final_message=ask, patterns=patterns)
    assert not needs_clarification_followup(agent=ok, diff_patch="x", stats=some, final_message=ask, patterns=patterns)
    assert not needs_clarification_followup(agent=ok, diff_patch="", stats=zero,
                                            final_message="I won't refactor this. Could you stop?", patterns=patterns)


def test_effort_and_verify_helpers():
    assert effort_minutes("5min") == 5 and effort_minutes("1h30min") == 90 and effort_minutes("1d") == 480
    assert effort_minutes(None) == 0
    assert classify(0, False, "") == "pass"
    assert classify(1, False, "[ERROR] COMPILATION ERROR :") == "compile_error"
    assert classify(1, False, "There are test failures.") == "test_failure"
    assert classify(1, False, "something else") == "build_error"
    assert classify(124, True, "") == "timeout"
    assert classify(127, False, "bash: mvn: command not found") == "missing_tool"
    assert classify(2, False, "bash: -c: line 0: unexpected EOF while looking for matching `''") == "invalid_command"
    assert classify(1, False, "compileJava FAILED: Could not load from remote cache: Not in GZIP format") == "build_error"
    assert is_infrastructure_result({"outcome": "no_jdk"})
    assert is_infrastructure_result({"outcome": "build_error", "returncode": 127,
                                     "log_tail": "bash: mvn: command not found"})
    assert is_infrastructure_result({"outcome": "build_error", "returncode": 2,
                                     "log_tail": "unexpected EOF while looking for matching `''"})
    assert not is_infrastructure_result({"outcome": "compile_error"})
    turns = [{"turn": 1, "lines_total": 4}, {"turn": 2, "lines_total": 0}, {"turn": 3, "lines_total": 2}]
    assert versions_to_check(turns) == [0, 1, 2, 3]
    assert versions_to_check(turns[:1] + [{"turn": 2, "lines_total": 1}]) == [0, 1, 2]


# ---- repos and scope ------------------------------------------------------------
def test_concurrent_worktrees(upstream, repos):
    repos.ensure_mirror("demo", upstream["url"], commits=[upstream["human"]])
    errors: list[Exception] = []

    def make(n: int) -> None:
        try:
            import subprocess

            ws, baseline, tree_ok = repos.create_workspace(
                "demo", f"inst-{n}", upstream["before"], f"claude-exp/t-{n}")
            assert tree_ok and ws.name == "demo"
            log = subprocess.run(["git", "rev-list", "--all"], cwd=ws, capture_output=True, text=True)
            assert log.stdout.split() == [baseline]              # one commit, no history
            (ws / "f.txt").write_text(str(n))
            subprocess.run(["git", "add", "-A"], cwd=ws, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=ws, check=True)
            repos.publish_branch("demo", f"inst-{n}", f"claude-exp/t-{n}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=make, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert all(repos.branch_exists("demo", f"claude-exp/t-{n}") for n in range(8))
    tips = {repos.rev_parse("demo", f"claude-exp/t-{n}") for n in range(8)}
    assert len(tips) == 8
    assert repos.show_file("demo", "claude-exp/t-3", "f.txt") == b"3"
    for n in range(8):
        repos.remove_workspace(f"inst-{n}")
    assert not any((repos.worktrees_dir / f"inst-{n}").exists() for n in range(8))


# ---- turn-1 target label --------------------------------------------------------
FILE_WITH_OVERLOADS = """package demo;

public class Box {
    /** Adds one. */
    public int add(int a) {
        return add(a, 1);
    }

    @Deprecated
    public static <T> int add(final int a,
                              @Nullable int b) throws IllegalStateException {
        return a + b;
    }

    int size(List<Map<String, Integer>> rows, String... names) { return rows.size(); }

    static class Inner {
        void reset() { }
    }

    static class Other {
        void reset() { }
    }
}
"""


def test_target_label_forms():
    from rdp.target import count_declarations, method_header, target_label

    one = "    /** Adds one. */\n    public int add(int a) {\n        return add(a, 1);\n    }"
    two = ("    @Deprecated\n    public static <T> int add(final int a,\n"
           "                              @Nullable int b) throws IllegalStateException {\n        return a + b;\n    }")
    assert method_header(one, "add")[0] == ["int"]
    assert method_header(two, "add") == (["int", "int"], "public static <T> int add(final int a, @Nullable int b)")
    assert count_declarations(FILE_WITH_OVERLOADS, "add", ["int"]) == 1          # the call add(a, 1) is not one
    assert count_declarations(FILE_WITH_OVERLOADS, "add", ["int", "int"]) == 1
    assert target_label("demo.Box#add", one, FILE_WITH_OVERLOADS) == ("Box.add(int)", "signature")
    assert target_label("demo.Box#add", two, FILE_WITH_OVERLOADS) == ("Box.add(int, int)", "signature")
    size = "    int size(List<Map<String, Integer>> rows, String... names) { return rows.size(); }"
    assert target_label("demo.Box#size", size, FILE_WITH_OVERLOADS) == (
        "Box.size(List<Map<String,Integer>>, String...)", "signature")
    # the same signature in two nested classes cannot be told apart: no label
    assert target_label("demo.Box.Inner#reset", "        void reset() { }", FILE_WITH_OVERLOADS) == ("", "")
    assert target_label("demo.Box", one, FILE_WITH_OVERLOADS) == ("", "")


def test_every_selected_instance_has_a_unique_label():
    records = load_dataset(load_settings().dataset)
    ambiguous: list = []
    pool = select_instances(records, ambiguous=ambiguous)
    assert pool and all(i.target_label and i.label_form in ("signature", "declaration") for i in pool)
    assert all(not i.target_label for i in ambiguous)
    assert len(pool) > 5 * len(ambiguous)                                         # the fix keeps most of the pool


def test_probe_counts():
    from rdp.runner.claude_parse import probe_counts

    def tool(name, **inp):
        import json as _json

        return _json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": name, "input": inp}]}})

    text = "\n".join([
        tool("Bash", command="git log --all --oneline | head"),
        tool("Bash", command="cd src && git show HEAD~1:Foo.java"),
        tool("Bash", command="git status && git diff"),
        tool("Bash", command="mvn -q test"),
        tool("WebFetch", url="https://example.org"),
        tool("Bash", command="curl -s https://example.org"),
        tool("Read", file_path="Foo.java"),
    ])
    assert probe_counts(text) == {"git_history_calls": 2, "web_calls": 2}
    assert probe_counts("") == {"git_history_calls": 0, "web_calls": 0}


def test_scope_and_human_version(upstream, repos):
    repos.ensure_mirror("demo", upstream["url"], commits=[upstream["human"]])
    inst = make_instance(upstream, 1)
    turns = [{"turn": 1, "changed_paths": [TARGET, "notes.md", "src/main/java/demo/New.java"]}]
    scope = build_scope(inst, turns)
    assert scope == ["src/main/java/demo/Calc.java", "src/main/java/demo/New.java"]
    base = scope_contents(repos, inst, scope, commit=upstream["before"], baseline_sha=upstream["before"])
    human = scope_contents(repos, inst, scope, commit=None, baseline_sha=upstream["before"])
    assert list(base) == [TARGET] and b"int r = a + b" in base[TARGET]
    assert b"private int add" in human[TARGET]
    assert scope_hash(base) != scope_hash(human)
    assert scope_hash(base) == scope_hash(dict(base))
    assert HUMAN == "human"
