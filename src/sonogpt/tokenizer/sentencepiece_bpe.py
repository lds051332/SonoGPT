"""SentencePiece BPE wrapper with fixed SonoGPT control tokens."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2

from sonogpt.tokenizer.base import (
    BOS_TOKEN,
    EOS_TOKEN,
    INPUT_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    TARGET_TOKEN,
    TASK_EXTRACT_TOKEN,
    TASK_GENERATE_TOKEN,
    UNK_TOKEN,
)

BPE_TOKENIZER_VERSION = "1.0.0"
_USER_DEFINED_SYMBOLS = (
    TASK_EXTRACT_TOKEN,
    TASK_GENERATE_TOKEN,
    INPUT_TOKEN,
    TARGET_TOKEN,
)


class SentencePieceBPETokenizer:
    """Thin, validated adapter around a frozen SentencePiece model."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._processor = spm.SentencePieceProcessor(
            model_file=str(self.model_path)
        )
        if not self._processor:
            raise ValueError("could not load SentencePiece model")
        for token in SPECIAL_TOKENS:
            token_id = self._processor.piece_to_id(token)
            if (
                token_id < 0
                or self._processor.id_to_piece(token_id) != token
            ):
                raise ValueError(f"SentencePiece model is missing {token}")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        model_path: Path,
        vocab_size: int = 4096,
        character_coverage: float = 0.9995,
    ) -> "SentencePieceBPETokenizer":
        """Train deterministically from caller-provided training text."""

        text_rows = tuple(texts)
        if not text_rows:
            raise ValueError("at least one training text is required")
        if any("\n" in text or "\r" in text for text in text_rows):
            raise ValueError("training texts must each occupy one corpus line")
        if vocab_size < 320:
            raise ValueError("vocab_size must leave room for byte fallback pieces")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        spm.set_min_log_level(2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            corpus_path = temporary_path / "corpus.txt"
            corpus_path.write_text(
                "\n".join(text_rows) + "\n", encoding="utf-8"
            )
            model_prefix = temporary_path / "sonogpt_bpe"
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                model_type="bpe",
                vocab_size=vocab_size,
                character_coverage=character_coverage,
                byte_fallback=True,
                normalization_rule_name="identity",
                add_dummy_prefix=False,
                remove_extra_whitespaces=False,
                split_by_whitespace=False,
                split_digits=True,
                pad_id=0,
                pad_piece=PAD_TOKEN,
                bos_id=1,
                bos_piece=BOS_TOKEN,
                eos_id=2,
                eos_piece=EOS_TOKEN,
                unk_id=3,
                unk_piece=UNK_TOKEN,
                user_defined_symbols=list(_USER_DEFINED_SYMBOLS),
                hard_vocab_limit=False,
                shuffle_input_sentence=False,
                num_threads=1,
            )
            model_proto = sentencepiece_model_pb2.ModelProto()
            model_proto.ParseFromString(
                model_prefix.with_suffix(".model").read_bytes()
            )
            model_proto.trainer_spec.ClearField("input")
            model_proto.trainer_spec.ClearField("model_prefix")
            model_path.write_bytes(
                model_proto.SerializeToString(deterministic=True)
            )
        return cls(model_path)

    @property
    def vocab_size(self) -> int:
        return self._processor.get_piece_size()

    def token_id(self, token: str) -> int:
        token_id = self._processor.piece_to_id(token)
        if token_id < 0 or self._processor.id_to_piece(token_id) != token:
            raise ValueError(f"token is not in the vocabulary: {token}")
        return token_id

    @property
    def pad_id(self) -> int:
        return self.token_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.token_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.token_id(EOS_TOKEN)

    @property
    def unk_id(self) -> int:
        return self.token_id(UNK_TOKEN)

    @property
    def task_generate_id(self) -> int:
        return self.token_id(TASK_GENERATE_TOKEN)

    @property
    def input_id(self) -> int:
        return self.token_id(INPUT_TOKEN)

    @property
    def target_id(self) -> int:
        return self.token_id(TARGET_TOKEN)

    def encode(self, text: str) -> list[int]:
        return list(self._processor.encode(text, out_type=int))

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        ids = list(token_ids)
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError("token ID is out of range")
        if skip_special_tokens:
            special_ids = {self.token_id(token) for token in SPECIAL_TOKENS}
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self._processor.decode(ids)

    @property
    def model_sha256(self) -> str:
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()
