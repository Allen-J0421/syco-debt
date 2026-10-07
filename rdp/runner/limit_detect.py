"""Provider usage/session-limit detection for coding-agent experiment turns.

Pure and stdlib-only (``json``, ``re``, ``datetime``, ``zoneinfo``) with no I/O, so the
classification logic is unit-testable in isolation.

Why this exists: a provider limit usually does *not* fail the CLI process. A Claude session
limit comes back as a normal ``stream-json`` ``result`` line with ``is_error:true`` /
``api_error_status:429`` and exit code 0, so the experiment keeps running and burns its
remaining turns on empty diffs. Without detection that dead run is recorded as ``done`` and
never re-run. This module classifies such a turn so the runner can abort *before* persisting
corrupt data; the orchestrator then pauses until the limit resets and re-runs the experiment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Limit kinds.
SESSION_LIMIT = "session_limit"
SPEND_CAP = "spend_cap"
QUOTA = "quota"
OVERLOAD = "overload"


@dataclass(frozen=True)
class LimitHit:
    """A detected provider limit for one experiment turn."""

    provider: str               # "claude" | "codex" | "gemini"
    reset_dt: datetime | None   # tz-aware reset time, or None when unparseable
    raw: str                    # the matched message (for logging)
    kind: str = SESSION_LIMIT   # one of SESSION_LIMIT/SPEND_CAP/QUOTA/OVERLOAD


@dataclass(frozen=True)
class LimitDetectConfig:
    """Detection knobs. Gemini surfacing is off by default to preserve the prompter's
    intentional silent fallback; only flip it on when you want quota errors to pause the run."""

    enabled: bool = True
    claude_patterns: tuple[re.Pattern[str], ...] = ()
    codex_patterns: tuple[re.Pattern[str], ...] = ()
    gemini_surface: bool = False        # surface Gemini 429 (quota/credits) as a limit
    gemini_surface_503: bool = False    # surface Gemini 503 (transient overload) as a limit


class ProviderLimitError(RuntimeError):
    """Raised to abort an experiment when a provider usage limit is detected.

    Kept here (not in ``models``) so ``models`` needs no new imports.
    """

    def __init__(self, hit: LimitHit) -> None:
        super().__init__(f"provider limit ({hit.provider}/{hit.kind})")
        self.hit = hit


# Claude session-limit text, e.g. "You've hit your session limit · resets 9:50am (...)".
_CLAUDE_LIMIT_RE = re.compile(r"hit your (?:usage|session) limit", re.I)
# Reset clause: "resets 9:50am (America/Los_Angeles)" or "resets 3:30am" (no tz).
_RESET_RE = re.compile(
    r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b(?:\s*\(([^)]+)\))?",
    re.I,
)
# Codex provider limits. Grounded on real captured codex.jsonl error events:
#   {"type":"error","message":"You've hit your usage limit. Upgrade to Pro ... or try again at 4:21 PM."}
#   {"type":"error","message":"You hit your spend cap set by the owner of your workspace ..."}
# Both exit the CLI without writing a diff, so they must be detected or the run burns every
# turn on empty diffs and (worse) cascades to the next model on the same provider.
_CODEX_LIMIT_RE = re.compile(
    r"hit your usage limit|hit your spend cap|spend[\s_-]?cap|"
    r"usage limit reached|purchase more credits|out of credits|"
    r"rate limit reached|too many requests",
    re.I,
)
# Spend-cap vs. usage-limit/quota classification (drives the marker's ``kind``).
_CODEX_SPENDCAP_RE = re.compile(r"spend[\s_-]?cap|turn\.failed.*spend cap", re.I)
_CODEX_QUOTA_RE = re.compile(r"usage limit|credits|rate limit|too many requests", re.I)
# Codex reset clause: "try again at 4:21 PM" (no timezone given -> assume local).
_CODEX_RESET_RE = re.compile(r"try again at\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b", re.I)


def _now_aware(now: datetime | None) -> datetime:
    """Return a tz-aware 'now' (default: local tz)."""
    if now is not None:
        return now if now.tzinfo is not None else now.astimezone()
    return datetime.now().astimezone()


def _iter_json_objects(text: str) -> Iterator[dict]:
    """Yield dict objects from JSONL text, skipping blank/malformed lines."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _resolve_walltime(
    hour: int,
    minute: int,
    ampm: str,
    tz_name: str | None,
    now: datetime | None,
) -> datetime | None:
    """Resolve a 12-hour wall-clock time to the next occurrence (today or tomorrow).

    Handles am/pm, noon/midnight, an optional timezone name (falls back to local on
    unknown/empty), and rolls forward a day if the time has already passed.
    """
    if not (1 <= hour <= 12) or minute > 59:
        return None
    # 12-hour -> 24-hour.
    if ampm == "am":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12

    tz = None
    if tz_name:
        try:
            tz = ZoneInfo(tz_name.strip())
        except (ZoneInfoNotFoundError, ValueError):
            tz = None

    base = _now_aware(now)
    if tz is not None:
        base = base.astimezone(tz)
    candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= base:
        candidate += timedelta(days=1)
    return candidate


def parse_claude_reset(text: str, *, now: datetime | None = None) -> datetime | None:
    """Parse a Claude reset clause into the next occurrence of that wall-clock time.

    Handles am/pm, optional minutes, optional timezone (falls back to local), noon/midnight,
    and rolls to tomorrow if the time has already passed today. Returns None on no/invalid match.
    """
    match = _RESET_RE.search(text)
    if not match:
        return None
    minute = int(match.group(2)) if match.group(2) else 0
    return _resolve_walltime(int(match.group(1)), minute, match.group(3).lower(),
                             match.group(4), now)


def parse_codex_reset(text: str, *, now: datetime | None = None) -> datetime | None:
    """Parse a Codex reset clause ("... try again at 4:21 PM") to the next wall-clock time.

    Codex omits a timezone, so the time resolves against the local zone. Returns None when
    no reset clause is present (the limit is still surfaced, just without a precise reset).
    """
    match = _CODEX_RESET_RE.search(text)
    if not match:
        return None
    minute = int(match.group(2)) if match.group(2) else 0
    return _resolve_walltime(int(match.group(1)), minute, match.group(3).lower(), None, now)


def detect_claude_limit(
    jsonl_text: str, cfg: LimitDetectConfig, *, now: datetime | None = None
) -> LimitHit | None:
    """Detect a Claude session/usage limit from claude.jsonl stream output."""
    patterns = (_CLAUDE_LIMIT_RE, *cfg.claude_patterns)
    for obj in _iter_json_objects(jsonl_text):
        if obj.get("type") != "result":
            continue
        result_text = obj.get("result")
        result_text = result_text if isinstance(result_text, str) else ""
        is_error = obj.get("is_error") is True
        api_status = obj.get("api_error_status")
        text_hit = any(p.search(result_text) for p in patterns)
        if (is_error and api_status == 429) or text_hit:
            raw = result_text or json.dumps(obj)[:300]
            return LimitHit(
                provider="claude",
                reset_dt=parse_claude_reset(result_text, now=now),
                raw=raw,
                kind=SESSION_LIMIT,
            )
    return None


def _codex_limit_kind(text: str) -> str:
    """Classify a matched Codex limit message: spend cap vs. usage-limit/quota.

    Unknown custom-pattern matches default to SPEND_CAP for backward compatibility.
    """
    if _CODEX_SPENDCAP_RE.search(text):
        return SPEND_CAP
    if _CODEX_QUOTA_RE.search(text):
        return QUOTA
    return SPEND_CAP


def detect_codex_limit(
    jsonl_text: str, cfg: LimitDetectConfig, *, now: datetime | None = None
) -> LimitHit | None:
    """Detect a Codex usage limit / spend cap from ``codex exec`` JSONL (stdout+stderr).

    Codex surfaces both as ``{"type":"error",...}`` / ``{"type":"turn.failed",...}`` events
    that exit the CLI without a diff. We match on the message text (which the Codex agent
    wrapper folds into ``jsonl_text``) and, when present, parse the "try again at ..." reset.
    """
    for pattern in (_CODEX_LIMIT_RE, *cfg.codex_patterns):
        match = pattern.search(jsonl_text)
        if match:
            start = max(0, match.start() - 60)
            raw = jsonl_text[start:match.end() + 160].strip()
            return LimitHit(
                provider="codex",
                reset_dt=parse_codex_reset(jsonl_text, now=now),
                raw=raw,
                kind=_codex_limit_kind(raw),
            )
    return None


def detect_gemini_limit(prompter_jsonl: str, cfg: LimitDetectConfig) -> LimitHit | None:
    """Detect a Gemini prompter quota/overload from already-logged prompter.jsonl error events.

    Returns None unless surfacing is explicitly enabled, preserving the prompter's silent
    fallback by default.
    """
    if not cfg.enabled or not (cfg.gemini_surface or cfg.gemini_surface_503):
        return None
    for obj in _iter_json_objects(prompter_jsonl):
        if obj.get("event") != "error":
            continue
        payload = obj.get("payload")
        status = payload.get("status_code") if isinstance(payload, dict) else None
        text = obj.get("text") if isinstance(obj.get("text"), str) else ""
        if status == 429 and cfg.gemini_surface:
            return LimitHit(provider="gemini", reset_dt=None, raw=text or "gemini 429", kind=QUOTA)
        if status == 503 and cfg.gemini_surface_503:
            return LimitHit(provider="gemini", reset_dt=None, raw=text or "gemini 503", kind=OVERLOAD)
    return None


def detect_limit(
    *,
    provider: str,
    jsonl_text: str,
    diff_patch: str = "",
    cfg: LimitDetectConfig,
    now: datetime | None = None,
) -> LimitHit | None:
    """Dispatch limit detection for a coding-agent turn by provider.

    ``diff_patch`` is accepted for symmetry/future heuristics; the primary signal is the
    JSONL error fields, not diff emptiness (an empty diff alone is a normal clarification case).
    Gemini lives in prompter.jsonl and is surfaced separately via ``detect_gemini_limit``.
    """
    if not cfg.enabled:
        return None
    if provider == "claude":
        return detect_claude_limit(jsonl_text, cfg, now=now)
    if provider == "codex":
        return detect_codex_limit(jsonl_text, cfg, now=now)
    return None
