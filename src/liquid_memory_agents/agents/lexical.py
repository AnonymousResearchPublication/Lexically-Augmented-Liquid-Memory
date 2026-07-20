from __future__ import annotations

from typing import Any, Protocol

import torch

from ..liquid import BoundedLexicalMemory
from .base import AgentResponse, BaseAgent


class Encoder(Protocol):
    dimension: int

    def encode(self, text: str | list[str]) -> torch.Tensor: ...


class Answerer(Protocol):
    def answer(
        self,
        query: str,
        context: str,
        query_timestamp: str | None = None,
    ) -> str: ...


class LexicalMemoryAgent(BaseAgent):
    """The LALM lexical sidecar without any Liquid state or virtual tokens."""

    name = "lexical_only"

    def __init__(
        self,
        encoder: Encoder,
        answerer: Answerer,
        capacity: int = 512,
        top_k: int = 8,
        max_text_bytes: int = 1024,
        redundancy_threshold: float = 0.82,
        correction_cues: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.encoder = encoder
        self.answerer = answerer
        self.memory = BoundedLexicalMemory(
            embedding_dim=encoder.dimension,
            capacity=capacity,
            top_k=top_k,
            max_text_bytes=max_text_bytes,
            redundancy_threshold=redundancy_threshold,
            correction_cues=correction_cues,
        )

    def reset(self) -> None:
        self.memory.reset()

    @staticmethod
    def _serialize(
        turn: str | dict[str, Any],
        timestamp: str | None,
        session_id: str | None,
    ) -> tuple[str, str]:
        if isinstance(turn, dict):
            role = str(turn.get("role", "user"))
            text = str(turn.get("content", ""))
        else:
            role, text = "user", str(turn)
        encoded_turn = (
            f"role={role}; timestamp={timestamp or 'unknown'}; content={text}"
        )
        return f"session={session_id or 'none'}; {encoded_turn}", encoded_turn

    def observe(
        self,
        turn: str | dict[str, Any],
        timestamp: str | None = None,
        session_id: str | None = None,
    ) -> None:
        stored_text, embedding_text = self._serialize(turn, timestamp, session_id)
        self.memory.update(stored_text, self.encoder.encode(embedding_text))

    def observe_many(
        self,
        observations: list[
            tuple[str | dict[str, Any], str | None, str | None]
        ],
    ) -> None:
        serialized = [
            self._serialize(turn, timestamp, session_id)
            for turn, timestamp, session_id in observations
        ]
        if serialized:
            stored_texts, embedding_texts = zip(*serialized)
            self.memory.update_many(
                list(stored_texts),
                self.encoder.encode(list(embedding_texts)),
            )

    def get_memory_context(self, query: str) -> tuple[str, dict[str, Any]]:
        context, metadata = self.memory.retrieve(self.encoder.encode(query))
        return context, {"latent_enabled": False, **metadata}

    def answer(
        self,
        query: str,
        query_timestamp: str | None = None,
    ) -> AgentResponse:
        context, metadata = self.get_memory_context(query)
        return AgentResponse(
            text=self.answerer.answer(query, context, query_timestamp),
            memory_context=context,
            metadata=metadata,
        )

    def memory_size_bytes(self) -> int:
        return self.memory.reserved_bytes

    def diagnostics(self) -> dict[str, Any]:
        return {"latent_enabled": False, **self.memory.diagnostics()}
