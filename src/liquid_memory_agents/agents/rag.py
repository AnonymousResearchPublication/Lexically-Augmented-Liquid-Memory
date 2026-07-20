from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .base import AgentResponse, BaseAgent


@dataclass
class TurnRecord:
    text: str
    embedding: torch.Tensor
    timestamp: str | None
    session_id: str


class RAGMemoryAgent(BaseAgent):
    """Growing-history, normalized inner-product, turn-to-session RAG."""

    name = "rag"

    def __init__(
        self, encoder, answerer, top_k: int = 8,
        max_turn_chars: int = 2000, max_context_chars: int = 12000,
    ) -> None:
        self.encoder, self.answerer, self.top_k = encoder, answerer, top_k
        self.max_turn_chars, self.max_context_chars = max_turn_chars, max_context_chars
        self.turns: list[TurnRecord] = []
        self.sessions: dict[str, list[str]] = {}
        self._last: dict[str, Any] = {}

    def reset(self) -> None:
        self.turns.clear()
        self.sessions.clear()
        self._last = {}

    def observe(self, turn, timestamp=None, session_id=None) -> None:
        text = turn.get("content", "") if isinstance(turn, dict) else str(turn)
        role = turn.get("role", "user") if isinstance(turn, dict) else "user"
        sid = session_id or f"turn-{len(self.turns)}"
        formatted = f"[{timestamp or 'unknown'}] {role}: {text}"
        embedding = F.normalize(self.encoder.encode(formatted).float(), dim=-1)
        self.turns.append(TurnRecord(formatted, embedding.cpu(), timestamp, sid))
        self.sessions.setdefault(sid, []).append(formatted)

    def observe_many(self, observations) -> None:
        if not observations:
            return
        formatted_rows = []
        for offset, (turn, timestamp, session_id) in enumerate(observations):
            text = turn.get("content", "") if isinstance(turn, dict) else str(turn)
            role = turn.get("role", "user") if isinstance(turn, dict) else "user"
            sid = session_id or f"turn-{len(self.turns) + offset}"
            formatted = f"[{timestamp or 'unknown'}] {role}: {text}"
            formatted_rows.append((formatted, timestamp, sid))
        embeddings = F.normalize(
            self.encoder.encode([row[0] for row in formatted_rows]).float(), dim=-1
        )
        for (formatted, timestamp, sid), embedding in zip(formatted_rows, embeddings):
            self.turns.append(TurnRecord(formatted, embedding.cpu(), timestamp, sid))
            self.sessions.setdefault(sid, []).append(formatted)

    def get_memory_context(self, query: str) -> tuple[str, dict[str, Any]]:
        if not self.turns:
            return "", {"retrieved_session_ids": []}
        query_embedding = F.normalize(self.encoder.encode(query).float(), dim=-1).cpu()
        matrix = torch.stack([turn.embedding for turn in self.turns])
        k = min(self.top_k, len(self.turns))
        scores, indices = torch.topk(matrix @ query_embedding, k=k)
        session_ids = list(dict.fromkeys(self.turns[i].session_id for i in indices.tolist()))
        # Full-session expansion can bury the matched evidence beyond the LLM
        # context window. Supply bounded hit snippets, strongest nearest query.
        def clipped(record: TurnRecord) -> str:
            text = record.text
            if len(text) > self.max_turn_chars:
                half = max(1, (self.max_turn_chars - 20) // 2)
                text = text[:half] + "\n...[clipped]...\n" + text[-half:]
            return f"[Session id: {record.session_id}]\n{text}"

        blocks = [clipped(self.turns[index]) for index in reversed(indices.tolist())]
        context = "\n\n".join(blocks)
        if len(context) > self.max_context_chars:
            context = context[-self.max_context_chars:]
        self._last = {
            "retrieved_session_ids": session_ids,
            "turn_indices": indices.tolist(),
            "scores": scores.tolist(),
        }
        return context, dict(self._last)

    def answer(self, query: str, query_timestamp: str | None = None) -> AgentResponse:
        context, metadata = self.get_memory_context(query)
        return AgentResponse(self.answerer.answer(query, context, query_timestamp), context, metadata)

    def memory_size_bytes(self) -> int:
        embedding_bytes = sum(x.embedding.numel() * x.embedding.element_size() for x in self.turns)
        turn_text = sum(len(x.text.encode()) for x in self.turns)
        index_payload = 8 * len(self.turns)
        # ``sessions`` stores references to the same formatted strings held by
        # ``turns``. Count the logical text payload once rather than treating
        # the two lookup structures as duplicate serialized copies.
        return embedding_bytes + turn_text + index_payload

    def diagnostics(self) -> dict[str, Any]:
        return {"turns": len(self.turns), "sessions": len(self.sessions), **self._last}
