"""Atomic checkpoint I/O and random-state capture for exact recovery."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

CHECKPOINT_VERSION = "1.0.0"
LATEST_POINTER_VERSION = "1.0.0"


def capture_random_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_random_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]

    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise ValueError("invalid CUDA random state")
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA random-state recovery")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(cuda_states)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_semantic_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray:")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping:{")
        for key in sorted(
            value,
            key=lambda item: (type(item).__name__, repr(item)),
        ):
            _update_semantic_digest(digest, key)
            _update_semantic_digest(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, tuple):
        digest.update(b"tuple:[")
        for item in value:
            _update_semantic_digest(digest, item)
        digest.update(b"]")
    elif isinstance(value, list):
        digest.update(b"list:[")
        for item in value:
            _update_semantic_digest(digest, item)
        digest.update(b"]")
    elif isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b":")
        digest.update(repr(value).encode("utf-8"))
    else:
        raise TypeError(
            f"unsupported checkpoint value for semantic digest: {type(value)}"
        )


def semantic_checkpoint_sha256(path: Path) -> str:
    """Hash checkpoint values independent of torch.save container metadata."""

    digest = hashlib.sha256()
    _update_semantic_digest(digest, load_checkpoint(path))
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: Mapping[str, object]) -> str:
    """Atomically write a complete trusted-local PyTorch checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    complete_payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        **payload,
    }
    try:
        torch.save(complete_payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return sha256_path(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted local checkpoint; arbitrary pickle files are unsafe."""

    payload = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    return payload


def write_latest_pointer(
    directory: Path, checkpoint_path: Path, checkpoint_sha256: str
) -> Path:
    directory = Path(directory)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.resolve() != directory.resolve():
        raise ValueError("latest checkpoint must be inside its run directory")
    pointer_path = directory / "latest.json"
    temporary_path = directory / "latest.json.tmp"
    payload = {
        "pointer_version": LATEST_POINTER_VERSION,
        "checkpoint": checkpoint_path.name,
        "sha256": checkpoint_sha256,
    }
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, pointer_path)
    return pointer_path


def resolve_latest_pointer(directory: Path) -> Path:
    directory = Path(directory)
    payload = json.loads(
        (directory / "latest.json").read_text(encoding="utf-8")
    )
    if payload.get("pointer_version") != LATEST_POINTER_VERSION:
        raise ValueError("unsupported latest-pointer version")
    checkpoint_path = directory / payload["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if sha256_path(checkpoint_path) != payload.get("sha256"):
        raise ValueError("latest checkpoint SHA-256 mismatch")
    return checkpoint_path
