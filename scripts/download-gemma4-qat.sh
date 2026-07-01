#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-unsloth/gemma-4-26B-A4B-it-qat-GGUF}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma-4-26B-A4B-it-qat-GGUF}"

python3 -m pip install --user --upgrade huggingface_hub hf_transfer
mkdir -p "$MODEL_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO"   --local-dir "$MODEL_DIR"   --include "*mmproj-BF16*"   --include "*UD-Q4_K_XL*"

echo "Downloaded $MODEL_REPO to $MODEL_DIR"
