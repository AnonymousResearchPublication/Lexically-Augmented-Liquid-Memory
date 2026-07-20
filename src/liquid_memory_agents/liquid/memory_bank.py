from __future__ import annotations

import torch
from torch import Tensor, nn

from .cell import AdaptiveLiquidCell


class LiquidMemoryBank(nn.Module):
    """Fixed-capacity persistent memory; parameters are model, not user memory."""

    def __init__(self, cell: AdaptiveLiquidCell, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.cell, self.storage_dtype = cell, dtype
        self.register_buffer(
            "state", torch.zeros(cell.slots, cell.state_dim, dtype=dtype), persistent=False
        )
        self._last: dict[str, Tensor] = {}
        self.update_count = 0

    @torch.no_grad()
    def reset(self) -> None:
        self.state.zero_()
        self._last = {}
        self.update_count = 0

    def update(self, embedding: Tensor) -> Tensor:
        next_state, diagnostics = self.cell(
            self.state.to(dtype=embedding.dtype, device=embedding.device), embedding
        )
        self.state = next_state.detach().to(dtype=self.storage_dtype)
        self._last = {k: v.detach().cpu() for k, v in diagnostics.items()}
        self.update_count += 1
        return next_state

    @property
    def persistent_bytes(self) -> int:
        return self.state.numel() * self.state.element_size()

    def diagnostics(self) -> dict[str, object]:
        return {
            "updates": self.update_count,
            "state_norm": float(self.state.float().norm().item()),
            "slot_norms": self.state.float().norm(dim=-1).cpu().tolist(),
            "dtype": str(self.storage_dtype).replace("torch.", ""),
            "persistent_bytes": self.persistent_bytes,
            **{k: v.numpy().tolist() for k, v in self._last.items()},
        }
