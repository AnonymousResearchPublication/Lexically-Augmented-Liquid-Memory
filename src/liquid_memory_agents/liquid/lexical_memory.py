from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


class BoundedLexicalMemory:
    """Fixed-capacity exact-text sidecar with query-independent writes."""

    CORRECTION_CUES = (
        "correction:", "actually", "instead", "not ", "now ", "changed",
        "currently", "update:",
    )

    def __init__(
        self,
        embedding_dim: int,
        capacity: int = 128,
        top_k: int = 8,
        max_text_bytes: int = 1024,
        redundancy_threshold: float = 0.82,
        correction_cues: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if capacity < 1 or top_k < 1 or max_text_bytes < 32:
            raise ValueError("invalid lexical-memory capacity, top_k, or text budget")
        self.embedding_dim = int(embedding_dim)
        self.capacity = int(capacity)
        self.top_k = int(top_k)
        self.max_text_bytes = int(max_text_bytes)
        self.redundancy_threshold = float(redundancy_threshold)
        self.correction_cues = tuple(correction_cues or self.CORRECTION_CUES)
        if not self.correction_cues or any(not cue for cue in self.correction_cues):
            raise ValueError("correction_cues must contain non-empty strings")
        self.embeddings = torch.zeros(self.capacity, self.embedding_dim)
        self.texts = [""] * self.capacity
        self.count = 0
        self._last: dict[str, Any] = {}

    def reset(self) -> None:
        self.embeddings.zero_()
        self.texts = [""] * self.capacity
        self.count = 0
        self._last = {}

    def _bounded_text(self, text: str) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= self.max_text_bytes:
            return text
        half = max(1, (self.max_text_bytes - 24) // 2)
        return (
            encoded[:half].decode("utf-8", errors="ignore")
            + "\n...[bounded]...\n"
            + encoded[-half:].decode("utf-8", errors="ignore")
        )

    def update(self, text: str, embedding: Tensor) -> None:
        vector = F.normalize(embedding.detach().float().cpu(), dim=-1)
        text = self._bounded_text(text)
        if self.count < self.capacity:
            index = self.count
            self.count += 1
        else:
            active = self.embeddings[: self.count]
            similarity = active @ vector
            closest = int(similarity.argmax())
            is_correction = any(cue in text.casefold() for cue in self.correction_cues)
            if float(similarity[closest]) >= self.redundancy_threshold:
                if not is_correction:
                    return
                index = closest
            else:
                pairwise = active @ active.T
                pairwise.fill_diagonal_(-1.0)
                index = int(pairwise.max(dim=1).values.argmax())
        self.embeddings[index].copy_(vector)
        self.texts[index] = text

    def update_many(self, texts: list[str], embeddings: Tensor) -> None:
        for text, embedding in zip(texts, embeddings):
            self.update(text, embedding)

    def retrieve(self, query_embedding: Tensor) -> tuple[str, dict[str, Any]]:
        if self.count == 0:
            self._last = {"lexical_entries": 0, "lexical_indices": [], "lexical_scores": []}
            return "", dict(self._last)
        query = F.normalize(query_embedding.detach().float().cpu(), dim=-1)
        scores, indices = torch.topk(
            self.embeddings[: self.count] @ query,
            k=min(self.top_k, self.count),
        )
        retrieved_texts = [self.texts[index] for index in indices.tolist()]
        blocks = [
            f"[Lexical memory entry {index}]\n{self.texts[index]}"
            for index in reversed(indices.tolist())
        ]
        session_ids = []
        for text in retrieved_texts:
            if text.startswith("session="):
                session_id = text[len("session="):].split(";", 1)[0]
                if session_id not in session_ids:
                    session_ids.append(session_id)
        self._last = {
            "lexical_entries": self.count,
            "lexical_capacity": self.capacity,
            "lexical_indices": indices.tolist(),
            "lexical_scores": scores.tolist(),
            "retrieved_session_ids": session_ids,
        }
        return "\n\n".join(blocks), dict(self._last)

    @property
    def reserved_bytes(self) -> int:
        return self.capacity * (self.embedding_dim * 4 + self.max_text_bytes)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "lexical_entries": self.count,
            "lexical_capacity": self.capacity,
            "lexical_reserved_bytes": self.reserved_bytes,
            "correction_cues": list(self.correction_cues),
            **self._last,
        }
