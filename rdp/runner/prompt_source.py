"""Where each turn's user message comes from.

Turn 1 is a fixed template that names the target (file and method, no refactoring type).
Turns 2..N come from the simulated novice developer, as in the foundational experiment.
"""

from __future__ import annotations

from typing import Protocol

from rdp.runner.models import PrompterConfig, PrompterTurnInput, PromptTurn
from rdp.runner.prompter import GeminiPrompter


def render_turn1(template: str, *, file: str, method: str) -> str:
    return template.strip().format(file=file, method=method)


class PromptSource(Protocol):
    def prepare(self, *, target_path: str, target_text: str) -> None: ...

    def next_prompt(self, turn: int, turn_input: PrompterTurnInput | None) -> PromptTurn: ...

    def respond_to_clarification(self, turn: int, coding_reply: str) -> PromptTurn: ...


class FixedPromptSource:
    """Template on turn 1, then the same fixed prompt every turn (no simulated user).

    Not the experimental condition: for dry runs and tests only.
    """

    def __init__(self, first_request: str, prompt: str) -> None:
        self._first = first_request
        self._prompt = prompt

    def prepare(self, *, target_path: str, target_text: str) -> None:
        return None

    def next_prompt(self, turn: int, turn_input: PrompterTurnInput | None) -> PromptTurn:
        if turn == 1:
            return PromptTurn(prompt=self._first, source="template")
        return PromptTurn(prompt=self._prompt, source="fixed")

    def respond_to_clarification(self, turn: int, coding_reply: str) -> PromptTurn:
        return PromptTurn(prompt=self._prompt, source="fixed")


class GeminiPromptSource:
    """Template on turn 1, Gemini novice user afterwards."""

    def __init__(self, first_request: str, config: PrompterConfig) -> None:
        self._first = first_request
        self._prompter = GeminiPrompter(config)

    def prepare(self, *, target_path: str, target_text: str) -> None:
        self._prompter.prime(target_path=target_path, target_text=target_text, first_request=self._first)

    def next_prompt(self, turn: int, turn_input: PrompterTurnInput | None) -> PromptTurn:
        if turn == 1:
            return PromptTurn(prompt=self._first, source="template")
        prompt = self._prompter.next_request(turn_input, turn=turn)
        return PromptTurn(prompt=prompt, prompter_jsonl=self._prompter.events_jsonl_for_turn(turn),
                          fallback=self._prompter.last_was_fallback, source="prompter")

    def respond_to_clarification(self, turn: int, coding_reply: str) -> PromptTurn:
        before = len(self._prompter.events)
        prompt = self._prompter.respond_to_clarification(coding_reply, turn=turn)
        import json

        events = "".join(json.dumps(ev, ensure_ascii=False) + "\n" for ev in self._prompter.events[before:])
        return PromptTurn(prompt=prompt, prompter_jsonl=events,
                          fallback=self._prompter.last_was_fallback, source="prompter")
