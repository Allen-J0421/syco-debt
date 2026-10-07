"""One instance, end to end: worktree, multi-turn agent session, per-turn log and artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rdp.instances import Instance
from rdp.repos import RepoCache
from rdp.runner.artifacts import append_clarification_reply, append_prompt_turn, write_step_artifacts
from rdp.runner.claude_parse import extract_final_agent_message, probe_counts
from rdp.runner.git_repo import GitRepository
from rdp.runner.iteration import run_iteration
from rdp.runner.limit_detect import LimitDetectConfig, ProviderLimitError
from rdp.runner.models import (
    CSV_COLUMNS,
    ClarificationPatterns,
    PrompterTurnInput,
    PromptTurn,
    agent_run_ok,
)
from rdp.runner.prompt_source import PromptSource, render_turn1
from rdp.state import InstanceState, instance_dir, save_state
from rdp.util import eprint, sanitize_slug, utc_iso, utc_stamp


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    repos: RepoCache
    model: str
    turns: int
    turn1_template: str
    prompter_label: str                                   # slug used in the prompt.txt file name
    agent_factory: Callable[[Path], object]               # work_dir -> agent with .run()
    prompt_source_factory: Callable[[str], PromptSource]  # first request -> source
    clarification_patterns: ClarificationPatterns | None
    limit_cfg: LimitDetectConfig
    keep_worktree: bool = False


@dataclass(frozen=True)
class RunPaths:
    inst_dir: Path
    stamp: str
    model_slug: str
    prompter_label: str

    @property
    def run_stamp(self) -> str:
        return f"{self.stamp}-{self.model_slug}"

    @property
    def log_csv(self) -> Path:
        return self.inst_dir / "logs" / f"{self.run_stamp}-log.csv"

    @property
    def turns_jsonl(self) -> Path:
        return self.inst_dir / "logs" / f"{self.run_stamp}-turns.jsonl"

    @property
    def prompt_txt(self) -> Path:
        return self.inst_dir / "logs" / f"{self.stamp}-{self.prompter_label}_{self.model_slug}_prompt.txt"

    @property
    def artifacts_dir(self) -> Path:
        return self.inst_dir / self.run_stamp

    def remove_outputs(self) -> None:
        for path in (self.log_csv, self.turns_jsonl, self.prompt_txt):
            path.unlink(missing_ok=True)
        shutil.rmtree(self.artifacts_dir, ignore_errors=True)


def branch_name(stamp: str, model: str, instance_id: str) -> str:
    return f"claude-exp/{stamp}-{sanitize_slug(model)}-{instance_id}"


def discard_run(ctx: RunContext, inst: Instance, state: InstanceState, *, reason: str) -> None:
    """Remove every trace of the current attempt so the instance can run fresh."""
    if state.stamp:
        RunPaths(instance_dir(ctx.run_dir, inst.instance_id), state.stamp,
                 sanitize_slug(ctx.model), ctx.prompter_label).remove_outputs()
    ctx.repos.remove_workspace(inst.instance_id)
    if state.branch:
        ctx.repos.delete_branch(inst.project, state.branch)   # present only if it had been published
        state.discarded.append({"stamp": state.stamp, "branch": state.branch, "reason": reason})
    state.stamp = state.branch = state.baseline_sha = ""
    state.turns_done = state.agent_failures = state.prompter_fallbacks = 0


def run_instance(ctx: RunContext, inst: Instance, state: InstanceState) -> InstanceState:
    """Run all turns for one instance. Raises ProviderLimitError after discarding the attempt."""
    stamp = utc_stamp()
    paths = RunPaths(instance_dir(ctx.run_dir, inst.instance_id), stamp,
                     sanitize_slug(ctx.model), ctx.prompter_label)
    branch = branch_name(stamp, ctx.model, inst.instance_id)

    state.status, state.stamp, state.branch = "running", stamp, branch
    state.attempts += 1
    state.started_at, state.finished_at, state.error = utc_iso(), "", ""
    state.turns_done = state.agent_failures = state.prompter_fallbacks = 0
    save_state(ctx.run_dir, state)

    try:
        parent = ctx.repos.parent_of(inst.project, inst.commit_id)
        # A standalone repository with one commit: no history for the agent to read the answer from.
        worktree, prev_sha, tree_ok = ctx.repos.create_workspace(
            inst.project, inst.instance_id, parent, branch)
        git = GitRepository(worktree)
        state.baseline_sha, state.baseline_tree_matches = prev_sha, tree_ok
        save_state(ctx.run_dir, state)

        first_request = render_turn1(ctx.turn1_template, file=inst.target_file,
                                     target=inst.target_label or inst.method)
        source = ctx.prompt_source_factory(first_request)
        source.prepare(target_path=inst.target_file, target_text=git.file_text(prev_sha, inst.target_file))
        agent = ctx.agent_factory(worktree)

        paths.log_csv.parent.mkdir(parents=True, exist_ok=True)
        session_id: str | None = None
        last_input: PrompterTurnInput | None = None

        with paths.log_csv.open("w", newline="", encoding="utf-8") as csv_file, \
                paths.turns_jsonl.open("w", encoding="utf-8") as turns_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_COLUMNS)

            for i in range(1, ctx.turns + 1):
                turn: PromptTurn = source.next_prompt(i, last_input)

                def responder(reply: str, _i: int = i) -> PromptTurn:
                    return source.respond_to_clarification(_i, reply)

                result = run_iteration(
                    git, agent,
                    iteration=i, prompt=turn.prompt, session_id=session_id, previous_sha=prev_sha,
                    target_file=inst.target_file, limit_cfg=ctx.limit_cfg,
                    clarification_patterns=ctx.clarification_patterns,
                    clarification_responder=responder,
                )

                writer.writerow(result.as_csv_row(model=ctx.model, branch=branch))
                csv_file.flush()
                append_prompt_turn(paths.prompt_txt, i, turn.prompt)
                if result.clarification_prompt_text:
                    append_clarification_reply(paths.prompt_txt, i, result.clarification_prompt_text)
                write_step_artifacts(
                    paths.artifacts_dir, i,
                    diff_patch=result.diff_patch, jsonl_text=result.jsonl_text,
                    agent_jsonl_name="claude.jsonl",
                    prompter_jsonl=turn.prompter_jsonl + result.clarification_prompter_jsonl,
                )
                fallback = turn.fallback or result.clarification_fallback
                record = {
                    "instance_id": inst.instance_id, "turn": i,
                    "commit_sha": result.commit_sha, "tree_sha": result.tree_sha,
                    "parent_sha": prev_sha,
                    **result.stats.as_dict(),
                    **result.java_stats.as_dict("java_"),
                    **result.target_stats.as_dict("target_"),
                    "changed_paths": list(result.changed_paths),
                    "duration_s": round(result.agent.duration_s, 3),
                    "exit_code": result.agent.exit_code, "timed_out": int(result.agent.timed_out),
                    "clarification_followup": int(result.clarification_followup),
                    "prompt_source": turn.source, "prompter_fallback": int(fallback),
                    "prompt": turn.prompt,
                    **probe_counts(result.jsonl_text),
                    **{f"usage_{k}": v for k, v in result.usage.items()},
                }
                turns_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                turns_file.flush()

                ok = agent_run_ok(result.agent)
                state.turns_done = i
                state.agent_failures += 0 if ok else 1
                state.prompter_fallbacks += int(fallback)
                save_state(ctx.run_dir, state)
                eprint(f"[{inst.instance_id}] turn {i:02d} +{result.stats.lines_added} "
                       f"-{result.stats.lines_deleted} exit={result.agent.exit_code}"
                       + (" (clarification follow-up)" if result.clarification_followup else "")
                       + (" (prompter fallback)" if fallback else ""))

                if session_id is None:
                    session_id = result.agent.session_id
                reply = extract_final_agent_message(result.jsonl_text) or None
                last_input = PrompterTurnInput(coding_reply=reply, diff_patch=result.diff_patch or None)
                prev_sha = result.commit_sha

    except ProviderLimitError:
        discard_run(ctx, inst, state, reason="provider_limit")
        raise
    except Exception as exc:
        state.status, state.error, state.finished_at = "failed", f"{type(exc).__name__}: {exc}", utc_iso()
        save_state(ctx.run_dir, state)
        try:                                   # keep whatever was committed, for diagnosis
            ctx.repos.publish_branch(inst.project, inst.instance_id, branch)
        except Exception:
            pass
        if not ctx.keep_worktree:
            ctx.repos.remove_workspace(inst.instance_id)
        return state

    try:
        ctx.repos.publish_branch(inst.project, inst.instance_id, branch)
    except Exception as exc:
        state.status, state.error, state.finished_at = "failed", f"publish failed: {exc}", utc_iso()
        save_state(ctx.run_dir, state)
        return state                           # workspace is kept so the commits are not lost
    if not ctx.keep_worktree:
        ctx.repos.remove_workspace(inst.instance_id)
    if state.agent_failures >= ctx.turns:
        state.status, state.error = "failed", "every turn failed (agent did not exit successfully)"
    else:
        state.status = "done"
    state.finished_at = utc_iso()
    save_state(ctx.run_dir, state)
    return state
