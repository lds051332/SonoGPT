"""Run the 32-sample character-tokenizer overfitting diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from sonogpt.data.dataset import (
    collate_encoded_samples,
    encode_generate_sample,
)
from sonogpt.data.semantic_generator import DEFAULT_SEED, sample_semantic_cases
from sonogpt.data.split import build_generate_splits
from sonogpt.model.config import SonoGPTConfig
from sonogpt.model.gpt import SonoGPT
from sonogpt.data.renderers import BASELINE_TEMPLATE_FAMILY
from sonogpt.tokenizer.character import CharacterTokenizer
from sonogpt.training.overfit import run_overfit
from sonogpt.training.reproducibility import set_reproducible_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "overfit" / "character_32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 32 <= args.sample_count <= 128:
        raise ValueError("sample-count must be between 32 and 128")
    set_reproducible_seed(args.seed)

    semantic_case_count = max(50, math.ceil(args.sample_count / 0.8))
    splits = build_generate_splits(
        sample_semantic_cases(semantic_case_count, args.seed),
        seed=args.seed,
    )
    selected_samples = tuple(
        sample
        for sample in splits.train
        if sample.template_family == BASELINE_TEMPLATE_FAMILY
    )[: args.sample_count]
    if len(selected_samples) != args.sample_count:
        raise RuntimeError("not enough unique semantic cases for overfitting")
    tokenizer = CharacterTokenizer.train(
        text
        for sample in splits.train
        for text in (sample.input, sample.target)
    )
    encoded_samples = tuple(
        encode_generate_sample(
            sample, tokenizer, max_seq_len=args.max_seq_len
        )
        for sample in selected_samples
    )
    batch = collate_encoded_samples(
        encoded_samples, pad_id=tokenizer.pad_id
    )
    config = SonoGPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        dropout=0.0,
    )
    model = SonoGPT(config)
    result = run_overfit(
        model,
        batch,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer_file_hash = tokenizer.save(
        args.output_directory / "character_tokenizer.json"
    )
    config.save(args.output_directory / "model_config.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "tokenizer_content_sha256": tokenizer.content_sha256,
            "sample_ids": [sample.sample_id for sample in selected_samples],
            "seed": args.seed,
            "steps": args.steps,
        },
        args.output_directory / "model.pt",
    )
    report = {
        **result.to_dict(),
        "device": args.device,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "max_sequence_tokens": int(batch.attention_mask.sum(dim=1).max()),
        "tokenizer_file_sha256": tokenizer_file_hash,
        "tokenizer_content_sha256": tokenizer.content_sha256,
    }
    (args.output_directory / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
