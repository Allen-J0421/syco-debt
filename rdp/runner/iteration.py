"""One turn: run the coding agent, stage, measure the change, check for a provider limit, commit."""

from __future__ import annotations

from collections.abc import Callable

from rdp.runner.claude_parse import extract_final_agent_message, extract_usage
from rdp.runner.clarification import merge_agent_results, needs_clarification_followup
from rdp.runner.git_repo import GitRepository, sum_stats
from rdp.runner.limit_detect import LimitDetectConfig, ProviderLimitError, detect_limit
from rdp.runner.models import (
    ClarificationPatterns,
    IterationResult,
    PromptTurn,
    agent_run_ok,
)

ClarificationResponder = Callable[[str], PromptTurn]


def _is_java(path: str) -> bool:
    return path.endswith(".java")


def run_iteration(
    git: GitRepository,
    agent,
    *,
    iteration: int,
    prompt: str,
    session_id: str | None,
    previous_sha: str,
    target_file: str,
    limit_cfg: LimitDetectConfig,
    clarification_patterns: ClarificationPatterns | None = None,
    clarification_responder: ClarificationResponder | None = None,
) -> IterationResult:
    is_first_turn = iteration == 1 and session_id is None

    agent_result, jsonl_text = agent.run(
        prompt, is_first=is_first_turn, session_id=None if is_first_turn else session_id
    )
    resume_session_id = agent_result.session_id or session_id

    def measure():
        git.stage_all()
        per_path = git.staged_numstat(previous_sha)
        return per_path, git.staged_diff(previous_sha)

    def check_limit(text: str) -> None:
        # Before any commit: a limited turn exits 0 with an empty diff and must never be recorded.
        hit = detect_limit(provider="claude", jsonl_text=text, cfg=limit_cfg)
        if hit is not None:
            git.unstage_and_restore(previous_sha)
            raise ProviderLimitError(hit)

    check_limit(jsonl_text)
    per_path, diff_patch = measure()
    stats = sum_stats(per_path)
    final_msg = extract_final_agent_message(jsonl_text)

    followup = False
    followup_turn: PromptTurn | None = None
    if (
        clarification_patterns is not None
        and clarification_responder is not None
        and resume_session_id
        and needs_clarification_followup(
            agent=agent_result, diff_patch=diff_patch, stats=stats,
            final_message=final_msg, patterns=clarification_patterns,
        )
    ):
        followup = True
        followup_turn = clarification_responder(final_msg)
        agent_result2, jsonl_text2 = agent.run(
            followup_turn.prompt, is_first=False, session_id=resume_session_id
        )
        check_limit(jsonl_text2)
        agent_result = merge_agent_results(agent_result, agent_result2)
        jsonl_text = jsonl_text + ("\n" if jsonl_text and jsonl_text2 else "") + jsonl_text2
        per_path, diff_patch = measure()
        stats = sum_stats(per_path)

    commit_message = f"claude-exp: iteration {iteration}"
    if followup:
        commit_message += " [clarification follow-up]"
    if not agent_run_ok(agent_result):
        commit_message += (
            f" [claude failed exit={agent_result.exit_code} timed_out={int(agent_result.timed_out)}]"
        )
    commit_sha = git.commit(commit_message)

    return IterationResult(
        number=iteration,
        stats=stats,
        java_stats=sum_stats(per_path, _is_java),
        target_stats=sum_stats(per_path, lambda p: p == target_file),
        agent=agent_result,
        commit_sha=commit_sha,
        commit_message=commit_message,
        tree_sha=git.tree_sha(),
        jsonl_text=jsonl_text,
        diff_patch=diff_patch,
        changed_paths=tuple(sorted(per_path)),
        clarification_prompt_text=followup_turn.prompt if followup_turn else "",
        clarification_followup=followup,
        clarification_prompter_jsonl=followup_turn.prompter_jsonl if followup_turn else "",
        clarification_fallback=bool(followup_turn and followup_turn.fallback),
        usage=extract_usage(jsonl_text),
    )
