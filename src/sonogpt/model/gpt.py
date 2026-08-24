"""Readable decoder-only Transformer used by SonoGPT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sonogpt.model.block import TransformerBlock
from sonogpt.model.config import SonoGPTConfig

IGNORE_INDEX = -100


@dataclass(frozen=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class SonoGPT(nn.Module):
    def __init__(self, config: SonoGPTConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_len, config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_seq_len {self.config.max_seq_len}"
            )
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids")
            attention_mask = attention_mask.bool()
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels shape must match input_ids")

        positions = torch.arange(
            sequence_length, device=input_ids.device
        ).unsqueeze(0)
        hidden_states = self.token_embeddings(
            input_ids
        ) + self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss = None
        if labels is not None:
            shifted_labels = labels[:, 1:].contiguous()
            if not torch.any(shifted_labels != IGNORE_INDEX):
                raise ValueError("labels contain no target tokens")
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int,
    ) -> torch.Tensor:
        """Greedy generation for unpadded prompts."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        generated = input_ids
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_seq_len:
                break
            next_logits = self(generated).logits[:, -1, :]
            next_ids = torch.argmax(next_logits, dim=-1)
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, eos_id),
                next_ids,
            )
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            finished |= next_ids == eos_id
            if bool(torch.all(finished)):
                break
        return generated

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        parameters = (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )
        return sum(parameter.numel() for parameter in parameters)
