"""Deterministic character tokenizer with reversible UTF-8 byte fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from sonogpt.tokenizer.base import (
    BOS_TOKEN,
    EOS_TOKEN,
    INPUT_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    TARGET_TOKEN,
    TASK_GENERATE_TOKEN,
    UNK_TOKEN,
)

CHARACTER_TOKENIZER_VERSION = "1.0.0"
_BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))


class CharacterTokenizer:
    """One-token-per-known-character baseline with lossless OOV handling."""

    def __init__(self, vocabulary: Sequence[str], *, byte_fallback: bool = True):
        self._vocabulary = tuple(vocabulary)
        self.byte_fallback = byte_fallback
        if self._vocabulary[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("vocabulary must begin with the fixed special tokens")
        if len(self._vocabulary) != len(set(self._vocabulary)):
            raise ValueError("vocabulary contains duplicate tokens")
        if byte_fallback:
            start = len(SPECIAL_TOKENS)
            if self._vocabulary[start : start + 256] != _BYTE_TOKENS:
                raise ValueError("byte fallback vocabulary is incomplete or reordered")

        self._token_to_id = {
            token: token_id for token_id, token in enumerate(self._vocabulary)
        }
        self._byte_token_ids = {
            self._token_to_id[token]: value
            for value, token in enumerate(_BYTE_TOKENS)
            if token in self._token_to_id
        }

    @classmethod
    def train(
        cls, texts: Iterable[str], *, byte_fallback: bool = True
    ) -> "CharacterTokenizer":
        """Build a stable vocabulary from training text only."""

        text_rows = tuple(texts)
        if not text_rows:
            raise ValueError("at least one training text is required")
        characters = sorted({character for text in text_rows for character in text})
        prefix = SPECIAL_TOKENS + (_BYTE_TOKENS if byte_fallback else ())
        return cls(prefix + tuple(characters), byte_fallback=byte_fallback)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._vocabulary

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    def token_id(self, token: str) -> int:
        try:
            return self._token_to_id[token]
        except KeyError as exc:
            raise ValueError(f"token is not in the vocabulary: {token}") from exc

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
        token_ids: list[int] = []
        for character in text:
            known_id = self._token_to_id.get(character)
            if known_id is not None:
                token_ids.append(known_id)
            elif self.byte_fallback:
                token_ids.extend(
                    self._token_to_id[_BYTE_TOKENS[value]]
                    for value in character.encode("utf-8")
                )
            else:
                token_ids.append(self.unk_id)
        return token_ids

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        pieces: list[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if not pending_bytes:
                return
            try:
                pieces.append(pending_bytes.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("token IDs contain invalid UTF-8 fallback bytes") from exc
            pending_bytes.clear()

        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID is out of range: {token_id}")
            if token_id in self._byte_token_ids:
                pending_bytes.append(self._byte_token_ids[token_id])
                continue

            flush_bytes()
            token = self._vocabulary[token_id]
            if token in SPECIAL_TOKENS and skip_special_tokens:
                continue
            pieces.append(token)

        flush_bytes()
        return "".join(pieces)

    def to_dict(self) -> dict[str, object]:
        return {
            "tokenizer_type": "character",
            "version": CHARACTER_TOKENIZER_VERSION,
            "byte_fallback": self.byte_fallback,
            "vocabulary": list(self._vocabulary),
        }

    @property
    def content_sha256(self) -> str:
        serialized = json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        path.write_text(serialized + "\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("tokenizer_type") != "character":
            raise ValueError("not a character tokenizer file")
        if payload.get("version") != CHARACTER_TOKENIZER_VERSION:
            raise ValueError("unsupported character tokenizer version")
        return cls(
            payload["vocabulary"],
            byte_fallback=bool(payload["byte_fallback"]),
        )
