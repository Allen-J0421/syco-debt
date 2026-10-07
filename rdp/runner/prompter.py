"""Gemini user agent: the simulated novice developer for turns 2..N.

Ported from the foundational harness. Differences:
- turn 1 is a fixed template naming the target, so the chat is primed with the target file,
  that first request, and the agent's reply before the prompter writes its first request;
- failures are retried, and a fallback to the fixed prompt is reported to the caller;
- calls go through a shared semaphore so parallel instances do not flood the API.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from rdp.runner.models import PrompterConfig, PrompterTurnInput
from rdp.util import eprint, non_empty_string, utc_iso

CODEBASE_HEADER = "The current codebase:"
FIRST_REQUEST_HEADER = "Your first request to the coding agent was:"
DIFF_HEADER = "Changes from the last iteration:"

_API_GATE = threading.BoundedSemaphore(4)


def set_max_concurrent(n: int) -> None:
    global _API_GATE
    _API_GATE = threading.BoundedSemaphore(max(1, int(n)))


def _part_records(response: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in response.candidates or []:
        content = candidate.content
        if content is None or not content.parts:
            continue
        for part in content.parts:
            record: dict[str, Any] = {}
            text = non_empty_string(part.text)
            if text:
                record["text"] = text
            if part.thought is not None:
                record["thought"] = part.thought
            sig = non_empty_string(part.thought_signature)
            if sig:
                record["thought_signature"] = sig
            if record:
                records.append(record)
    return records


def _parse_response(response: Any) -> dict[str, Any]:
    parts = _part_records(response)
    answer_parts = [str(r["text"]) for r in parts if r.get("text") and r.get("thought") is not True]
    thought_parts = [str(r["text"]) for r in parts if r.get("text") and r.get("thought") is True]
    finish_reason: str | None = None
    if response.candidates:
        reason = response.candidates[0].finish_reason
        if reason is not None:
            finish_reason = reason.name if hasattr(reason, "name") else str(reason)
    parsed: dict[str, Any] = {
        "answer_text": "\n".join(answer_parts),
        "thought_text": "\n".join(thought_parts),
        "parts": parts,
        "finish_reason": finish_reason,
        "response_id": response.response_id,
        "model_version": response.model_version,
    }
    if response.usage_metadata is not None:
        parsed["usage_metadata"] = response.usage_metadata.model_dump(mode="json", exclude_none=True)
    if response.prompt_feedback is not None:
        parsed["prompt_feedback"] = response.prompt_feedback.model_dump(mode="json", exclude_none=True)
    return parsed


@dataclass
class GeminiPrompter:
    """Stateful Gemini user agent with lossless event logging. One per instance."""

    config: PrompterConfig
    turn: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    last_was_fallback: bool = False
    _preamble: str = ""
    _client: Any | None = field(default=None, init=False, repr=False)
    _chat: Any | None = field(default=None, init=False, repr=False)

    def prime(self, *, target_path: str, target_text: str, first_request: str) -> None:
        """Context sent ahead of the prompter's first message: the code and the turn-1 request."""
        parts = []
        if target_text.strip():
            parts.append(f"{CODEBASE_HEADER}\n\n=== {target_path} ===\n{target_text.rstrip()}\n")
        parts.append(f"{FIRST_REQUEST_HEADER}\n\n{first_request.strip()}")
        self._preamble = "\n\n".join(parts)

    def _ensure_chat(self) -> Any:
        if self._chat is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self.config.api_key)
            self._chat = self._client.chats.create(
                model=self.config.model,
                config=types.GenerateContentConfig(system_instruction=self.config.system_prompt),
            )
        return self._chat

    def _log_event(self, event: str, *, text: str | None = None, payload: dict | None = None) -> None:
        entry: dict[str, Any] = {"ts": utc_iso(), "turn": self.turn, "agent": "prompter", "event": event}
        if text is not None:
            entry["text"] = text
        if payload is not None:
            entry["payload"] = payload
        self.events.append(entry)

    def _with_preamble(self, body: str) -> str:
        if self._preamble:
            body = f"{self._preamble}\n\n{body}"
            self._preamble = ""
        return body

    def _build_user_text(self, turn_input: PrompterTurnInput | None) -> str:
        parts: list[str] = []
        reply = non_empty_string(turn_input.coding_reply) if turn_input else None
        if reply:
            parts.append(reply)
        diff = non_empty_string(turn_input.diff_patch) if turn_input else None
        if diff:
            parts.append(f"{DIFF_HEADER}\n\n```diff\n{diff}\n```")
        parts.append(self.config.nudge)
        return self._with_preamble("\n\n".join(parts))

    def _log_chat_history(self, chat: Any) -> None:
        try:
            history = chat.get_history(curated=True)
        except Exception:
            return
        self._log_event("chat_history", payload={
            "curated": [c.model_dump(mode="json", exclude_none=True) for c in history],
            "message_count": len(history),
        })

    def _send(self, user_text: str, *, phase: str) -> str | None:
        """Send one message with retries. Returns the reply text, or None after final failure."""
        chat = self._ensure_chat()
        attempts = max(1, self.config.retries)
        for attempt in range(1, attempts + 1):
            try:
                with _API_GATE:
                    response = chat.send_message(user_text)
            except Exception as exc:  # APIError and transport errors alike
                code = getattr(exc, "code", None)
                err_text = non_empty_string(getattr(exc, "message", None)) or str(exc)
                self._log_event("error", text=err_text, payload={
                    "status_code": code, "body": getattr(exc, "details", None),
                    "phase": phase, "attempt": attempt,
                })
                if attempt < attempts:
                    time.sleep(min(60, 5 * 2 ** (attempt - 1)))
                    continue
                self._log_chat_history(chat)
                eprint(f"[prompter] turn {self.turn}: {phase} failed after {attempts} attempts: {err_text}")
                return None
            parsed = _parse_response(response)
            reply = non_empty_string(parsed["answer_text"]) or non_empty_string(response.text)
            self._log_event(
                "clarification_response" if phase == "clarification" else "response",
                text=reply or "",
                payload={"raw": response.model_dump(mode="json", exclude_none=True), "parsed": parsed},
            )
            self._log_chat_history(chat)
            return reply
        return None

    def _finish(self, reply: str | None, *, out_event: str) -> str:
        if not reply:
            self.last_was_fallback = True
            reply = self.config.fallback_prompt
        else:
            self.last_was_fallback = False
        self._log_event(out_event, text=reply, payload={"fallback": self.last_was_fallback})
        return reply

    def next_request(self, turn_input: PrompterTurnInput | None, *, turn: int) -> str:
        self.turn = turn
        user_text = self._build_user_text(turn_input)
        self._log_event("request", payload={
            "model": self.config.model,
            "system_instruction": self.config.system_prompt,
            "message": user_text,
        })
        return self._finish(self._send(user_text, phase="request"), out_event="prompt_out")

    def respond_to_clarification(self, coding_reply: str, *, turn: int) -> str:
        """Reply to a mid-turn clarification question from the coding agent."""
        self.turn = turn
        reply = non_empty_string(coding_reply) or ""
        user_text = self._with_preamble(f"{reply}\n\n{self.config.clarification_nudge}".strip())
        self._log_event("clarification_request", payload={"message": user_text, "coding_reply": reply})
        return self._finish(self._send(user_text, phase="clarification"),
                            out_event="clarification_prompt_out")

    def events_jsonl_for_turn(self, turn: int) -> str:
        lines = [json.dumps(ev, ensure_ascii=False) for ev in self.events if int(ev.get("turn", -1)) == turn]
        return "\n".join(lines) + ("\n" if lines else "")
