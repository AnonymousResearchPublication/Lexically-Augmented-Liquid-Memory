from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class TrainingExample:
    turn_embeddings: Tensor
    query_embedding: Tensor
    prompt_ids: Tensor
    answer_ids: Tensor
    answer_embedding: Tensor | None = None
    stale_answer_embedding: Tensor | None = None


class LiquidTrainer:
    """Teacher-forced prefix training on a split disjoint from evaluation."""

    def __init__(
        self, cell, reader, adapter, language_model, optimizer,
        diversity_weight=0.01, correction_weight=0.1, alignment_weight=0.1,
    ):
        self.cell, self.reader, self.adapter = cell, reader, adapter
        self.language_model, self.optimizer = language_model, optimizer
        self.diversity_weight = diversity_weight
        self.correction_weight = correction_weight
        self.alignment_weight = alignment_weight
        for parameter in self.language_model.parameters():
            parameter.requires_grad_(False)

    def loss(self, example: TrainingExample) -> tuple[Tensor, dict[str, float]]:
        memory = torch.zeros(
            self.cell.slots,
            self.cell.state_dim,
            device=example.turn_embeddings.device,
        )
        for turn_embedding in example.turn_embeddings:
            memory, _ = self.cell(memory, turn_embedding)
        read, _ = self.reader(example.query_embedding, memory)
        prefix = self.adapter(read).unsqueeze(0)
        ids = torch.cat((example.prompt_ids, example.answer_ids), dim=-1)
        token_embeddings = self.language_model.get_input_embeddings()(ids)
        # Frozen local LLMs commonly load in bfloat16/float16 while the trainable
        # memory modules remain float32. Cast only at the LLM boundary; autograd
        # still propagates through this cast into the prefix adapter.
        prefix = prefix.to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        )
        inputs = torch.cat((prefix, token_embeddings), dim=1)
        ignore_prefix = prefix.shape[1] + example.prompt_ids.shape[1]
        labels = torch.cat(
            (
                torch.full((1, ignore_prefix), -100, device=ids.device, dtype=torch.long),
                example.answer_ids,
            ),
            dim=1,
        )
        outputs = self.language_model(inputs_embeds=inputs, labels=labels)
        generation_loss = outputs.loss
        shifted_labels = labels[:, 1:]
        predictions = outputs.logits[:, :-1].argmax(dim=-1)
        answer_mask = shifted_labels.ne(-100)
        token_accuracy = (
            predictions.eq(shifted_labels)[answer_mask].float().mean()
            if answer_mask.any()
            else torch.ones((), device=memory.device)
        )
        exact = (
            predictions.eq(shifted_labels).logical_or(~answer_mask).all(dim=-1).float().mean()
        )
        normalized = F.normalize(memory, dim=-1)
        similarity = normalized @ normalized.T
        diversity = (similarity - torch.eye(self.cell.slots, device=similarity.device)).pow(2).mean()
        correction = torch.zeros((), device=memory.device)
        alignment = torch.zeros((), device=memory.device)
        if example.answer_embedding is not None:
            current_target = self.reader.query(example.answer_embedding.to(read.device))
            alignment = (1.0 - F.cosine_similarity(read, current_target, dim=-1)).mean()
        if example.answer_embedding is not None and example.stale_answer_embedding is not None:
            stale_target = self.reader.query(example.stale_answer_embedding.to(read.device))
            current = F.cosine_similarity(
                read, current_target, dim=-1
            )
            stale = F.cosine_similarity(
                read, stale_target, dim=-1
            )
            correction = F.relu(0.2 - current + stale).mean()
        total = (
            generation_loss
            + self.diversity_weight * diversity
            + self.alignment_weight * alignment
            + self.correction_weight * correction
        )
        return total, {
            "generation_loss": float(generation_loss.detach()),
            "slot_diversity_loss": float(diversity.detach()),
            "answer_alignment_loss": float(alignment.detach()),
            "correction_loss": float(correction.detach()),
            "token_accuracy": float(token_accuracy.detach()),
            "teacher_forced_exact": float(exact.detach()),
        }

    def step(self, example: TrainingExample) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.loss(example)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]], 1.0
        )
        self.optimizer.step()
        return {"loss": float(loss.detach()), **metrics}
