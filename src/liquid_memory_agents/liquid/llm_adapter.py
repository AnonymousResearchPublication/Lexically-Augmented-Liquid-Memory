from __future__ import annotations

import torch
from torch import Tensor, nn


class StateToPrefixAdapter(nn.Module):
    """Map a liquid read vector to trainable virtual tokens for a frozen answer LLM."""

    def __init__(self, state_dim: int, llm_embedding_dim: int, prefix_tokens: int = 8) -> None:
        super().__init__()
        self.prefix_tokens, self.llm_embedding_dim = prefix_tokens, llm_embedding_dim
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, prefix_tokens * llm_embedding_dim),
        )

    def forward(self, read_state: Tensor) -> Tensor:
        unbatched = read_state.ndim == 1
        if unbatched:
            read_state = read_state.unsqueeze(0)
        result = self.network(read_state).view(
            read_state.shape[0], self.prefix_tokens, self.llm_embedding_dim
        )
        return result.squeeze(0) if unbatched else result

    def generate(
        self,
        model: nn.Module,
        input_ids: Tensor,
        read_state: Tensor,
        attention_mask: Tensor | None = None,
        **generation_kwargs: object,
    ) -> Tensor:
        token_embeddings = model.get_input_embeddings()(input_ids)
        prefix = self(read_state)
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        inputs_embeds = torch.cat((prefix.to(token_embeddings.dtype), token_embeddings), dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_mask = torch.ones(
            input_ids.shape[0], self.prefix_tokens, dtype=attention_mask.dtype, device=input_ids.device
        )
        return model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat((prefix_mask, attention_mask), dim=1),
            **generation_kwargs,
        )
