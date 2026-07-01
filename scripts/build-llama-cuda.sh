#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"

sudo apt-get update
sudo apt-get install -y pciutils build-essential cmake curl libcurl4-openssl-dev git

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
else
  git -C "$LLAMA_CPP_DIR" pull --ff-only
fi

cmake "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"   -DBUILD_SHARED_LIBS=OFF   -DGGML_CUDA=ON

cmake --build "$LLAMA_CPP_DIR/build"   --config Release   -j   --clean-first   --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

cp "$LLAMA_CPP_DIR"/build/bin/llama-* "$LLAMA_CPP_DIR"/

echo "Built CUDA llama.cpp at $LLAMA_CPP_DIR"
