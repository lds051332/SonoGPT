from __future__ import annotations

from pathlib import Path

import torch

from sonogpt.model.config import SonoGPTConfig
from sonogpt.model.gpt import SonoGPT

CANDIDATE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "model"
    / "sonogpt_16m_candidate.json"
)


def _config() -> SonoGPTConfig:
    return SonoGPTConfig(
        vocab_size=64,
        max_seq_len=32,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )


def test_model_shapes_loss_and_weight_tying() -> None:
    torch.manual_seed(1)
    model = SonoGPT(_config())
    input_ids = torch.randint(0, 64, (2, 12))
    labels = input_ids.clone()
    labels[:, :7] = -100
    output = model(input_ids, labels=labels)

    assert output.logits.shape == (2, 12, 64)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()


def test_causal_mask_blocks_future_tokens() -> None:
    torch.manual_seed(2)
    model = SonoGPT(_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 40, 41]])

    first_logits = model(first).logits
    second_logits = model(second).logits

    torch.testing.assert_close(
        first_logits[:, :3], second_logits[:, :3], rtol=0.0, atol=1e-6
    )


def test_padding_does_not_change_valid_token_logits() -> None:
    torch.manual_seed(3)
    model = SonoGPT(_config()).eval()
    unpadded = torch.tensor([[1, 2, 3]])
    padded = torch.tensor([[1, 2, 3, 0, 0]])
    padding_mask = torch.tensor([[True, True, True, False, False]])

    expected = model(unpadded).logits
    actual = model(padded, attention_mask=padding_mask).logits[:, :3]

    torch.testing.assert_close(expected, actual, rtol=0.0, atol=1e-6)


def test_greedy_generation_stops_at_eos() -> None:
    model = SonoGPT(_config()).eval()
    for parameter in model.parameters():
        parameter.data.zero_()

    generated = model.generate(
        torch.tensor([[1, 2, 3]]),
        max_new_tokens=5,
        eos_id=0,
    )

    assert generated.tolist() == [[1, 2, 3, 0]]


def test_saved_state_restores_identical_logits(tmp_path: Path) -> None:
    torch.manual_seed(4)
    config = _config()
    model = SonoGPT(config).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = model(input_ids).logits
    state_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), state_path)

    restored = SonoGPT(config).eval()
    restored.load_state_dict(torch.load(state_path, weights_only=True))
    actual = restored(input_ids).logits

    torch.testing.assert_close(expected, actual)


def test_candidate_model_config_has_verified_parameter_count() -> None:
    config = SonoGPTConfig.load(CANDIDATE_CONFIG_PATH)
    model = SonoGPT(config)

    assert config.vocab_size == 1807
    assert config.max_seq_len == 384
    assert model.count_parameters() == 15_003_648
