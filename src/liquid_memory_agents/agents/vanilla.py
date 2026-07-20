from __future__ import annotations

from typing import Any

from .base import AgentResponse, BaseAgent


class VanillaAgent(BaseAgent):
    name = "vanilla"

    def __init__(self, answerer) -> None:
        self.answerer = answerer

    def reset(self) -> None: pass

    def observe(self, turn, timestamp=None, session_id=None) -> None: pass

    def get_memory_context(self, query: str) -> tuple[str, dict[str, Any]]:
        return "", {}

    def answer(self, query: str, query_timestamp: str | None = None) -> AgentResponse:
        return AgentResponse(self.answerer.answer(query, "", query_timestamp))

    def memory_size_bytes(self) -> int: return 0

    def diagnostics(self) -> dict[str, Any]: return {"memory_items": 0}
