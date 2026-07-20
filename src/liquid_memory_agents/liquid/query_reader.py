from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class QueryConditionedReader(nn.Module):
    def __init__(self, query_dim: int, state_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(query_dim, state_dim, bias=False)
        self.key = nn.Linear(state_dim, state_dim, bias=False)
        self.value = nn.Linear(state_dim, state_dim, bias=False)

    def forward(self, query: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        unbatched = memory.ndim == 2
        if unbatched:
            query, memory = query.unsqueeze(0), memory.unsqueeze(0)
        scores = torch.einsum("bd,bsd->bs", self.query(query), self.key(memory))
        weights = torch.softmax(scores / math.sqrt(memory.shape[-1]), dim=-1)
        read = torch.einsum("bs,bsd->bd", weights, self.value(memory))
        return (read.squeeze(0), weights.squeeze(0)) if unbatched else (read, weights)
