from __future__ import annotations

import hashlib
from typing import Any, Protocol

import torch

from ..liquid import (
    AdaptiveLiquidCell,
    BoundedLexicalMemory,
    LiquidMemoryBank,
    QueryConditionedReader,
    StateToPrefixAdapter,
)
from .base import AgentResponse, BaseAgent


class Encoder(Protocol):
    dimension: int

    def encode(self, text: str) -> torch.Tensor: ...


class PrefixAnswerer(Protocol):
    embedding_dimension: int

    def answer_with_prefix(
        self, query: str, prefix: torch.Tensor, query_timestamp: str | None = None,
        context: str = "",
    ) -> str: ...


class LALMMemoryAgent(BaseAgent):
    """Lexically Augmented Liquid Memory with two fixed-capacity pathways."""

    name = "lalm"

    PREFIX_CONTROLS = {
        "lalm": "learned",
        "pure_liquid": "learned",
        "lalm_zero_prefix": "zero",
        "lalm_random_prefix": "random",
        "lalm_permuted_prefix": "permuted",
    }

    def __init__(
        self,
        encoder: Encoder,
        answerer: PrefixAnswerer,
        state_dim: int = 256,
        slots: int = 8,
        prefix_tokens: int = 8,
        storage_dtype: torch.dtype = torch.float32,
        lexical_enabled: bool = True,
        lexical_capacity: int = 128,
        lexical_top_k: int = 8,
        lexical_max_text_bytes: int = 1024,
        lexical_redundancy_threshold: float = 0.82,
        lexical_prefix_scale: float = 0.1,
        lexical_correction_cues: tuple[str, ...] | list[str] | None = None,
        condition_name: str = "lalm",
        **cell_kwargs: Any,
    ) -> None:
        if condition_name not in self.PREFIX_CONTROLS:
            raise ValueError(f"unsupported LALM condition name: {condition_name}")
        self.name = condition_name
        self.prefix_control = self.PREFIX_CONTROLS[condition_name]
        self.encoder, self.answerer = encoder, answerer
        cell = AdaptiveLiquidCell(encoder.dimension, state_dim, slots, **cell_kwargs)
        self.memory = LiquidMemoryBank(cell, dtype=storage_dtype)
        self.reader = QueryConditionedReader(encoder.dimension, state_dim)
        self.adapter = StateToPrefixAdapter(
            state_dim, answerer.embedding_dimension, prefix_tokens=prefix_tokens
        )
        semantic_encoder = getattr(encoder, "semantic_encoder", None)
        self.lexical_memory = (
            BoundedLexicalMemory(
                semantic_encoder.dimension,
                lexical_capacity,
                lexical_top_k,
                lexical_max_text_bytes,
                lexical_redundancy_threshold,
                lexical_correction_cues,
            )
            if lexical_enabled and semantic_encoder is not None
            else None
        )
        self.lexical_prefix_scale = float(lexical_prefix_scale)
        self._last_read: dict[str, Any] = {}

    @staticmethod
    def _stable_seed(*parts: str) -> int:
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)

    def _controlled_prefix(self, prefix: torch.Tensor, query: str) -> torch.Tensor:
        """Apply sham-prefix controls while preserving lexical retrieval.

        The controls isolate whether LALM's gains come from the trained
        recurrent state or merely from prepending any continuous tokens.
        """
        if self.prefix_control == "learned":
            return prefix
        if self.prefix_control == "zero":
            return torch.zeros_like(prefix)
        if self.prefix_control == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._stable_seed(self.name, query))
            random_prefix = torch.randn(
                prefix.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(device=prefix.device, dtype=prefix.dtype)
            prefix_norm = prefix.detach().float().norm().clamp_min(1e-8)
            random_norm = random_prefix.detach().float().norm().clamp_min(1e-8)
            return random_prefix * (prefix_norm / random_norm).to(random_prefix.dtype)
        if self.prefix_control == "permuted":
            flat = prefix.reshape(-1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._stable_seed(self.name, str(flat.numel())))
            permutation = torch.randperm(flat.numel(), generator=generator).to(flat.device)
            return flat.index_select(0, permutation).reshape_as(prefix)
        raise AssertionError(f"unhandled prefix control: {self.prefix_control}")

    def reset(self) -> None:
        self.memory.reset()
        if self.lexical_memory is not None:
            self.lexical_memory.reset()
        self._last_read = {}

    def observe(
        self, turn: str | dict[str, Any], timestamp: str | None = None, session_id: str | None = None
    ) -> None:
        if isinstance(turn, dict):
            role, text = str(turn.get("role", "user")), str(turn.get("content", ""))
        else:
            role, text = "user", str(turn)
        encoded_turn = f"role={role}; timestamp={timestamp or 'unknown'}; content={text}"
        lexical_text = f"session={session_id or 'none'}; {encoded_turn}"
        if self.lexical_memory is not None:
            latent, semantic = self.encoder.encode_with_semantic(encoded_turn)
            self.memory.update(latent.float())
            self.lexical_memory.update(lexical_text, semantic)
        else:
            self.memory.update(self.encoder.encode(encoded_turn).float())

    def observe_many(
        self,
        observations: list[
            tuple[str | dict[str, Any], str | None, str | None]
        ],
    ) -> None:
        encoded_turns = []
        for turn, timestamp, _session_id in observations:
            if isinstance(turn, dict):
                role, text = str(turn.get("role", "user")), str(turn.get("content", ""))
            else:
                role, text = "user", str(turn)
            encoded_turns.append(
                f"role={role}; timestamp={timestamp or 'unknown'}; content={text}"
            )
        if not encoded_turns:
            return
        if self.lexical_memory is not None:
            latent_embeddings, semantic_embeddings = self.encoder.encode_with_semantic(
                encoded_turns
            )
            lexical_texts = [
                f"session={session_id or 'none'}; {encoded}"
                for encoded, (_, _, session_id) in zip(encoded_turns, observations)
            ]
            self.lexical_memory.update_many(lexical_texts, semantic_embeddings)
        else:
            latent_embeddings = self.encoder.encode(encoded_turns)
        for embedding in latent_embeddings.float():
            self.memory.update(embedding)

    def get_memory_context(self, query: str) -> tuple[str, dict[str, Any]]:
        if self.lexical_memory is not None:
            query_embedding, semantic_query = self.encoder.encode_with_semantic(query)
            context, lexical_metadata = self.lexical_memory.retrieve(semantic_query)
        else:
            query_embedding = self.encoder.encode(query)
            context, lexical_metadata = "", {}
        query_embedding = query_embedding.float()
        read, weights = self.reader(query_embedding, self.memory.state.float())
        prefix = self._controlled_prefix(self.adapter(read), query)
        self._last_read = {
            "attention": weights.detach().cpu().tolist(),
            "read_norm": float(read.detach().norm().item()),
            "prefix_control": self.prefix_control,
            "prefix": prefix,
        }
        return context, {
            **{k: v for k, v in self._last_read.items() if k != "prefix"},
            **lexical_metadata,
        }

    def answer(self, query: str, query_timestamp: str | None = None) -> AgentResponse:
        context, metadata = self.get_memory_context(query)
        text = self.answerer.answer_with_prefix(
            query,
            self._last_read["prefix"] * (
                self.lexical_prefix_scale if context else 1.0
            ),
            query_timestamp,
            context=context,
        )
        return AgentResponse(text=text, memory_context=context, metadata=metadata)

    def memory_size_bytes(self) -> int:
        lexical = self.lexical_memory.reserved_bytes if self.lexical_memory is not None else 0
        return self.memory.persistent_bytes + lexical

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.memory.diagnostics(),
            **(
                self.lexical_memory.diagnostics()
                if self.lexical_memory is not None
                else {"lexical_enabled": False}
            ),
            **{k: v for k, v in self._last_read.items() if k != "prefix"},
        }
