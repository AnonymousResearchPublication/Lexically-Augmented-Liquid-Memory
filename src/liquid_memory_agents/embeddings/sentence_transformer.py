from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class SentenceTransformerEncoder:
    """Normalized semantic embeddings; no hash fallback exists in primary code."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device=None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, text: str | list[str]) -> Tensor:
        values = self.model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        values = np.asarray(values, dtype=np.float32)
        return torch.from_numpy(values)


class LexicalAugmentedEncoder:
    """Semantic embeddings plus deterministic positional UTF-8 features.

    The lexical channel preserves spelling needed for unseen names and identifiers.
    It is bounded, non-learned, and is never used as a text store or retrieval index.
    """

    def __init__(self, semantic_encoder: SentenceTransformerEncoder, max_bytes: int = 256):
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.semantic_encoder = semantic_encoder
        self.max_bytes = int(max_bytes)
        self.dimension = semantic_encoder.dimension + self.max_bytes + 1

    def encode(self, text: str | list[str]) -> Tensor:
        result, _ = self.encode_with_semantic(text)
        return result

    def encode_with_semantic(self, text: str | list[str]) -> tuple[Tensor, Tensor]:
        scalar = isinstance(text, str)
        texts = [text] if scalar else list(text)
        semantic = self.semantic_encoder.encode(texts)
        lexical = torch.zeros((len(texts), self.max_bytes + 1), dtype=torch.float32)
        for row, value in enumerate(texts):
            encoded = value.encode("utf-8")[: self.max_bytes]
            if encoded:
                lexical[row, : len(encoded)] = torch.tensor(
                    list(encoded), dtype=torch.float32
                ) / 255.0
            lexical[row, -1] = len(encoded) / self.max_bytes
        result = torch.cat((semantic, lexical), dim=-1)
        if scalar:
            return result[0], semantic[0]
        return result, semantic
