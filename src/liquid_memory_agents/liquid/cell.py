from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdaptiveLiquidCell(nn.Module):
    """Input-dependent liquid update over a fixed bank of memory slots.

    For each slot, tau and write/overwrite gates depend on both the previous
    state and current semantic input. The query is never an input to this cell.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        slots: int,
        tau_min: float = 0.1,
        tau_max: float = 20.0,
        dt: float = 0.1,
        state_clip: float = 5.0,
        adaptive_tau: bool = True,
        update: str = "adaptive",
        memory_decay: float = 0.01,
    ) -> None:
        super().__init__()
        if slots < 1 or state_dim < 1:
            raise ValueError("slots and state_dim must be positive")
        self.input_dim, self.state_dim, self.slots = input_dim, state_dim, slots
        self.tau_min, self.tau_max, self.dt = tau_min, tau_max, dt
        self.state_clip, self.adaptive_tau = state_clip, adaptive_tau
        if update not in {"standard", "decay_enhanced", "adaptive"}:
            raise ValueError(f"unsupported update formulation: {update}")
        self.update_mode, self.memory_decay = update, memory_decay
        joint_dim = state_dim + input_dim
        self.tau_net = nn.Linear(joint_dim, 1)
        self.write_gate = nn.Linear(joint_dim, 1)
        self.overwrite_gate = nn.Linear(joint_dim, 1)
        self.recurrent = nn.Linear(state_dim, state_dim, bias=False)
        self.input_projection = nn.Linear(input_dim, state_dim)
        self.slot_keys = nn.Parameter(torch.empty(slots, state_dim))
        self.novelty_projection = nn.Linear(input_dim, state_dim, bias=False)
        nn.init.normal_(self.slot_keys, std=state_dim**-0.5)
        nn.init.orthogonal_(self.recurrent.weight, gain=0.9)

    def forward(self, memory: Tensor, inputs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        unbatched = memory.ndim == 2
        if unbatched:
            memory, inputs = memory.unsqueeze(0), inputs.unsqueeze(0)
        if memory.shape[1:] != (self.slots, self.state_dim):
            raise ValueError(f"expected memory [B,{self.slots},{self.state_dim}]")
        expanded = inputs[:, None, :].expand(-1, self.slots, -1)
        joint = torch.cat((memory, expanded), dim=-1)
        if self.adaptive_tau:
            tau = self.tau_min + F.softplus(self.tau_net(joint))
            tau = tau.clamp(max=self.tau_max)
        else:
            tau = torch.ones_like(self.tau_net(joint))
        learned_gate = torch.sigmoid(self.write_gate(joint))
        gate = torch.ones_like(learned_gate) if self.update_mode == "standard" else learned_gate
        overwrite = torch.sigmoid(self.overwrite_gate(joint))
        projected = F.normalize(self.novelty_projection(inputs), dim=-1)
        normalized_memory = F.normalize(memory, dim=-1)
        novelty = 1.0 - torch.einsum("bsd,bd->bs", normalized_memory, projected).abs()
        assignment_logits = torch.einsum("bd,sd->bs", projected, F.normalize(self.slot_keys, dim=-1))
        assignment_logits = assignment_logits + novelty
        assignment = torch.softmax(assignment_logits, dim=-1).unsqueeze(-1)
        target = torch.tanh(self.recurrent(memory) + self.input_projection(inputs)[:, None, :])
        derivative = (-memory + target) / tau
        if self.update_mode == "decay_enhanced":
            derivative = derivative - self.memory_decay * memory
        effective_gate = assignment * gate * (0.5 + 0.5 * overwrite * novelty.unsqueeze(-1))
        updated = memory + self.dt * effective_gate * derivative
        updated = torch.nan_to_num(updated).clamp(-self.state_clip, self.state_clip)
        diagnostics = {
            "tau": tau,
            "write_gate": gate,
            "overwrite_gate": overwrite,
            "assignment": assignment,
            "novelty": novelty,
        }
        if unbatched:
            return updated.squeeze(0), {k: v.squeeze(0) for k, v in diagnostics.items()}
        return updated, diagnostics
