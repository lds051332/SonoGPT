"""Versioned configuration for SonoGPT decoder-only models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_CONFIG_VERSION = "1.0.0"


@dataclass(frozen=True)
class SonoGPTConfig:
    vocab_size: int
    max_seq_len: int = 512
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields.values()
        ):
            raise ValueError("model dimensions must be positive integers")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, object]:
        return {"config_version": MODEL_CONFIG_VERSION, **asdict(self)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SonoGPTConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.pop("config_version", None) != MODEL_CONFIG_VERSION:
            raise ValueError("unsupported model config version")
        return cls(**payload)
