"""Shared tokenizer contract and SonoGPT control tokens."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
TASK_EXTRACT_TOKEN = "<task_extract>"
TASK_GENERATE_TOKEN = "<task_generate>"
INPUT_TOKEN = "<input>"
TARGET_TOKEN = "<target>"

SPECIAL_TOKENS = (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    TASK_EXTRACT_TOKEN,
    TASK_GENERATE_TOKEN,
    INPUT_TOKEN,
    TARGET_TOKEN,
)


@runtime_checkable
class SonoTokenizer(Protocol):
    """Minimum interface required by data encoding and inference."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def unk_id(self) -> int: ...

    @property
    def task_generate_id(self) -> int: ...

    @property
    def input_id(self) -> int: ...

    @property
    def target_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str: ...
