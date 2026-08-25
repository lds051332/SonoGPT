#!/usr/bin/env bash
# Install project deps with CPU PyTorch. Do not use plain `uv sync` on
# Linux/Windows: pyproject pins torch to the cu126 index.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --extra dev --no-install-package torch --inexact
uv pip install --reinstall "torch==2.12.0" --index-url https://download.pytorch.org/whl/cpu
uv run python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
