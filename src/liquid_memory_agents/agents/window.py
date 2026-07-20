from __future__ import annotations

from collections import deque
from typing import Any

from .base import AgentResponse, BaseAgent


class WindowMemoryAgent(BaseAgent):
    name = "window"

    def __init__(self, answerer, max_messages: int = 50) -> None:
        self.answerer, self.max_messages = answerer, max_messages
        self.messages: deque[str] = deque(maxlen=max_messages)

    def reset(self) -> None: self.messages.clear()

    def observe(self, turn, timestamp=None, session_id=None) -> None:
        text = turn.get("content", "") if isinstance(turn, dict) else str(turn)
        role = turn.get("role", "user") if isinstance(turn, dict) else "user"
        self.messages.append(f"[{timestamp or 'unknown'}][{session_id or 'none'}] {role}: {text}")

    def get_memory_context(self, query: str) -> tuple[str, dict[str, Any]]:
        return "\n".join(self.messages), {"window_messages": len(self.messages)}

    def answer(self, query: str, query_timestamp: str | None = None) -> AgentResponse:
        context, metadata = self.get_memory_context(query)
        return AgentResponse(self.answerer.answer(query, context, query_timestamp), context, metadata)

    def memory_size_bytes(self) -> int:
        return sum(len(item.encode("utf-8")) for item in self.messages)

    def diagnostics(self) -> dict[str, Any]: return {"window_messages": len(self.messages)}
