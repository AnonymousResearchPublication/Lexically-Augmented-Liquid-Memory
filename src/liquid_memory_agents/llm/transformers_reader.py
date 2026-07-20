from __future__ import annotations

import torch

from ..liquid.llm_adapter import StateToPrefixAdapter
from .prompts import answer_system_prompt, render_prompt


class TransformersAnswerReader:
    """One deterministic local answer model shared across all agent policies."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_input_tokens: int = 4096,
        max_new_tokens: int = 64,
        abstention_response: str = "I do not know.",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Memory precedes the question. Preserve the trailing question when the
        # complete prompt exceeds the configured context budget.
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype
        ).eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.system_prompt = answer_system_prompt(abstention_response)
        self.embedding_dimension = int(self.model.get_input_embeddings().embedding_dim)
        self.last_generation_hit_limit = False

    def _tokens(self, query: str, context: str, query_timestamp: str | None):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": render_prompt(query, context, query_timestamp)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        )
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in tokens.items()}

    def _batch_tokens(
        self,
        requests: list[tuple[str, str, str | None]],
    ):
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": render_prompt(query, context, query_timestamp),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for query, context, query_timestamp in requests
        ]
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in tokens.items()}

    @torch.inference_mode()
    def answer(self, query: str, context: str, query_timestamp: str | None = None) -> str:
        tokens = self._tokens(query, context, query_timestamp)
        output = self.model.generate(
            **tokens,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0, tokens["input_ids"].shape[1] :]
        self.last_generation_hit_limit = generated.shape[0] >= self.max_new_tokens
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def answer_batch(
        self,
        requests: list[tuple[str, str, str | None]],
    ) -> list[str]:
        if not requests:
            return []
        tokens = self._batch_tokens(requests)
        output = self.model.generate(
            **tokens,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_width = tokens["input_ids"].shape[1]
        generated = output[:, prompt_width:]
        self.last_generation_hit_limits = [
            not bool((row == self.tokenizer.eos_token_id).any().item())
            for row in generated
        ]
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in generated
        ]

    @torch.inference_mode()
    def answer_with_prefix(
        self, query: str, prefix: torch.Tensor, query_timestamp: str | None = None,
        context: str = "",
    ) -> str:
        tokens = self._tokens(query, context, query_timestamp)
        # Adapter.generate only needs these shape attributes; prefix is already adapted.
        token_embeddings = self.model.get_input_embeddings()(tokens["input_ids"])
        prefix = prefix.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        if prefix.ndim == 2:
            prefix = prefix.unsqueeze(0)
        inputs_embeds = torch.cat((prefix, token_embeddings), dim=1)
        prefix_mask = torch.ones(
            (tokens["input_ids"].shape[0], prefix.shape[1]),
            device=tokens["attention_mask"].device,
            dtype=tokens["attention_mask"].dtype,
        )
        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat((prefix_mask, tokens["attention_mask"]), dim=1),
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.last_generation_hit_limit = output.shape[1] >= self.max_new_tokens
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def answer_with_prefix_batch(
        self,
        requests: list[tuple[str, torch.Tensor, str | None, str]],
    ) -> list[str]:
        if not requests:
            return []
        tokens = self._batch_tokens([
            (query, context, query_timestamp)
            for query, _prefix, query_timestamp, context in requests
        ])
        token_embeddings = self.model.get_input_embeddings()(tokens["input_ids"])
        prefixes = torch.stack([
            prefix.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
            for _query, prefix, _timestamp, _context in requests
        ])
        inputs_embeds = torch.cat((prefixes, token_embeddings), dim=1)
        prefix_mask = torch.ones(
            (tokens["input_ids"].shape[0], prefixes.shape[1]),
            device=tokens["attention_mask"].device,
            dtype=tokens["attention_mask"].dtype,
        )
        output = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat((prefix_mask, tokens["attention_mask"]), dim=1),
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.last_generation_hit_limits = [
            not bool((row == self.tokenizer.eos_token_id).any().item())
            for row in output
        ]
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in output
        ]
