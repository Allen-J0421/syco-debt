"""Detect coding-agent clarification requests that warrant an in-iteration follow-up."""

from __future__ import annotations

from rdp.runner.models import (
    AgentRunResult,
    ClarificationPatterns,
    LineStats,
    agent_run_ok,
)


def _layer1_zero_change(
    agent: AgentRunResult,
    diff_patch: str,
    stats: LineStats,
) -> bool:
    return (
        agent_run_ok(agent)
        and stats.lines_total == 0
        and not diff_patch.strip()
    )


def _matches_any(patterns: tuple, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _layer2_clarification_keywords(
    final_message: str,
    patterns: ClarificationPatterns,
) -> bool:
    text = final_message.strip()
    if not text:
        return False
    if _matches_any(patterns.refusal_blockers, text):
        return False
    if patterns.require_question_mark and "?" not in text:
        return False
    return _matches_any(patterns.clarification_signals, text)


def needs_clarification_followup(
    *,
    agent: AgentRunResult,
    diff_patch: str,
    stats: LineStats,
    final_message: str,
    patterns: ClarificationPatterns,
) -> bool:
    if not _layer1_zero_change(agent, diff_patch, stats):
        return False
    return _layer2_clarification_keywords(final_message, patterns)


def merge_agent_results(
    first: AgentRunResult,
    second: AgentRunResult,
) -> AgentRunResult:
    """Combine two subprocess results from one logical iteration."""
    return AgentRunResult(
        exit_code=second.exit_code,
        duration_s=first.duration_s + second.duration_s,
        timed_out=second.timed_out,
        session_id=first.session_id or second.session_id,
    )
