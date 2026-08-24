"""Inspect the candidate SonoGPT model without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sonogpt.model.config import SonoGPTConfig
from sonogpt.model.gpt import SonoGPT
from sonogpt.tokenizer.sentencepiece_bpe import SentencePieceBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "model"
        / "sonogpt_16m_candidate.json",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tokenizers"
        / "sonogpt_bpe_1807_candidate_v2"
        / "sonogpt_bpe.model",
    )
    parser.add_argument("--smoke-seq-len", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "model_inspection"
        / "sonogpt_16m_candidate.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SonoGPTConfig.load(args.config)
    if not 1 <= args.smoke_seq_len <= config.max_seq_len:
        raise ValueError("smoke-seq-len must fit the configured context")

    tokenizer_hash = None
    if args.tokenizer_model.exists():
        tokenizer = SentencePieceBPETokenizer(args.tokenizer_model)
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("model and tokenizer vocabulary sizes do not match")
        tokenizer_hash = tokenizer.model_sha256

    torch.manual_seed(20260824)
    model = SonoGPT(config).eval()
    sample_input = torch.randint(
        0, config.vocab_size, (2, args.smoke_seq_len)
    )
    with torch.inference_mode():
        logits = model(sample_input).logits
    if logits.shape != (
        2,
        args.smoke_seq_len,
        config.vocab_size,
    ):
        raise RuntimeError("model smoke test returned an unexpected shape")
    if not torch.isfinite(logits).all():
        raise RuntimeError("model smoke test produced non-finite logits")

    parameter_count = model.count_parameters()
    report = {
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "parameter_count_millions": round(parameter_count / 1_000_000, 6),
        "parameter_memory_mib": {
            "fp32": round(parameter_count * 4 / 1024**2, 3),
            "fp16": round(parameter_count * 2 / 1024**2, 3),
        },
        "smoke_test": {
            "batch_size": 2,
            "sequence_length": args.smoke_seq_len,
            "logits_shape": list(logits.shape),
            "finite": True,
        },
        "tokenizer_model_sha256": tokenizer_hash,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
