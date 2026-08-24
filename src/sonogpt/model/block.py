"""Pre-LayerNorm Transformer block."""

from __future__ import annotations

import torch
from torch import nn

from sonogpt.model.attention import CausalSelfAttention
from sonogpt.model.config import SonoGPTConfig


class FeedForward(nn.Module):
    def __init__(self, config: SonoGPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config: SonoGPTConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(
            config.d_model, bias=config.bias
        )
        self.feed_forward = FeedForward(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states), attention_mask
        )
        return hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
