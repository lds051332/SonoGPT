"""Train and validate the candidate SonoGPT BPE tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from sonogpt.data.dataset import encode_generate_sample
from sonogpt.data.manifest import sha256_file, verify_manifest
from sonogpt.data.renderers import GeneratedSample
from sonogpt.tokenizer.sentencepiece_bpe import (
    BPE_TOKENIZER_VERSION,
    SentencePieceBPETokenizer,
)
from sonogpt.tokenizer.validation import validate_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = (
    "train",
    "validation",
    "test_seen_templates",
    "test_heldout_templates",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "synthetic_v1_5k_candidate_v2",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "manifests"
        / "synthetic_v1_5k_candidate_v2.manifest.json",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tokenizers"
        / "sonogpt_bpe_1807_candidate_v2",
    )
    parser.add_argument("--vocab-size", type=int, default=1807)
    parser.add_argument("--max-seq-len", type=int, default=384)
    return parser.parse_args()


def _load_samples(path: Path) -> tuple[GeneratedSample, ...]:
    return tuple(
        GeneratedSample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _sample_texts(samples: Iterable[GeneratedSample]) -> tuple[str, ...]:
    return tuple(
        text for sample in samples for text in (sample.input, sample.target)
    )


def _corpus_sha256(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _length_summary(lengths: list[int]) -> dict[str, int | float]:
    ordered = sorted(lengths)

    def percentile(percent: int) -> int:
        index = max(0, (percent * len(ordered) + 99) // 100 - 1)
        return ordered[index]

    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def _validate_split(
    tokenizer: SentencePieceBPETokenizer,
    samples: tuple[GeneratedSample, ...],
    *,
    max_seq_len: int,
) -> dict[str, object]:
    texts = _sample_texts(samples)
    validation = validate_tokenizer(tokenizer, texts)
    sequence_lengths = [
        len(
            encode_generate_sample(
                sample, tokenizer, max_seq_len=1_000_000
            ).input_ids
        )
        for sample in samples
    ]
    return {
        "sample_count": len(samples),
        "text_count": validation.text_count,
        "token_count": validation.token_count,
        "unknown_count": validation.unknown_count,
        "unknown_rate": validation.unknown_rate,
        "sequence_tokens": _length_summary(sequence_lengths),
        "over_max_seq_len_count": sum(
            length > max_seq_len for length in sequence_lengths
        ),
    }


def main() -> None:
    args = parse_args()
    verify_manifest(args.data_manifest, args.data_directory)
    train_samples = _load_samples(args.data_directory / "train.jsonl")
    training_texts = _sample_texts(train_samples)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    model_path = args.output_directory / "sonogpt_bpe.model"
    tokenizer = SentencePieceBPETokenizer.train(
        training_texts,
        model_path=model_path,
        vocab_size=args.vocab_size,
    )

    split_metrics = {
        split_name: _validate_split(
            tokenizer,
            _load_samples(args.data_directory / f"{split_name}.jsonl"),
            max_seq_len=args.max_seq_len,
        )
        for split_name in SPLIT_NAMES
    }
    report = {
        "tokenizer_type": "sentencepiece_bpe",
        "tokenizer_version": BPE_TOKENIZER_VERSION,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "vocab_size_matches_request": tokenizer.vocab_size == args.vocab_size,
        "byte_fallback": True,
        "normalization": "identity",
        "training_splits": ["train"],
        "training_sample_count": len(train_samples),
        "training_text_count": len(training_texts),
        "training_corpus_sha256": _corpus_sha256(training_texts),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "tokenizer_model_sha256": tokenizer.model_sha256,
        "max_seq_len": args.max_seq_len,
        "split_metrics": split_metrics,
    }
    report_path = args.output_directory / "tokenizer_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
