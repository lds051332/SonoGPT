from __future__ import annotations

import torch

from sonogpt.data.dataset import CausalLMBatch
from sonogpt.model.config import SonoGPTConfig
from sonogpt.model.gpt import SonoGPT
from sonogpt.training.overfit import run_overfit
from sonogpt.training.reproducibility import set_reproducible_seed


def test_overfit_diagnostic_reduces_target_loss() -> None:
    input_ids = torch.tensor(
        [
            [1, 4, 5, 6, 7, 2],
            [1, 4, 6, 5, 8, 2],
            [1, 5, 4, 7, 9, 2],
            [1, 6, 4, 8, 9, 2],
        ]
    )
    labels = input_ids.clone()
    labels[:, :3] = -100
    batch = CausalLMBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
    )
    config = SonoGPTConfig(
        vocab_size=16,
        max_seq_len=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    set_reproducible_seed(71)
    model = SonoGPT(config)

    result = run_overfit(
        model,
        batch,
        steps=60,
        batch_size=4,
        learning_rate=3e-3,
        seed=71,
    )

    assert result.final_loss < result.initial_loss * 0.25
    assert result.final_token_accuracy > result.initial_token_accuracy
