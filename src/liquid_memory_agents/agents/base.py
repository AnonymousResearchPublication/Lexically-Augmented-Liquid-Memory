from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentResponse:
    text: str
    memory_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Uniform interface used by every evaluated memory policy."""

    name = "base"

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def observe(
        self, turn: str | dict[str, Any], timestamp: str | None = None, session_id: str | None = None
    ) -> None: ...

    @abstractmethod
    def answer(self, query: str, query_timestamp: str | None = None) -> AgentResponse: ...

    @abstractmethod
    def get_memory_context(self, query: str) -> tuple[str, dict[str, Any]]: ...

    @abstractmethod
    def memory_size_bytes(self) -> int: ...

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]: ...
