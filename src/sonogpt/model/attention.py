"""Causal multi-head self-attention for SonoGPT."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sonogpt.model.config import SonoGPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SonoGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.qkv = nn.Linear(
            config.d_model, 3 * config.d_model, bias=config.bias
        )
        self.output = nn.Linear(
            config.d_model, config.d_model, bias=config.bias
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size, sequence_length, model_width = hidden_states.shape
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        dropout_probability = self.dropout if self.training else 0.0

        if attention_mask is None:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout_probability,
                is_causal=True,
            )
        else:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask shape must match input_ids")
            causal = torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril()
            allowed = causal[None, None, :, :] & attention_mask[
                :, None, None, :
            ].bool()
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=dropout_probability,
                is_causal=False,
            )

        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, model_width)
        )
        output = self.residual_dropout(self.output(attended))
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output
