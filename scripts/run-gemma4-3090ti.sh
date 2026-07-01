#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_CPP_DIR/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma-4-26B-A4B-it-qat-GGUF}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj-BF16.gguf}"
ALIAS="${ALIAS:-unsloth/gemma-4-26B-A4B-it-qat-GGUF}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-8192}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
  echo "Missing llama-server at $LLAMA_SERVER_BIN" >&2
  echo "Run: ./scripts/build-llama-cuda.sh" >&2
  exit 1
fi
if [ ! -f "$MODEL_FILE" ] || [ ! -f "$MMPROJ_FILE" ]; then
  echo "Missing Gemma 4 QAT model or mmproj under $MODEL_DIR" >&2
  echo "Run: ./scripts/download-gemma4-qat.sh" >&2
  exit 1
fi

cat <<EOF
Starting CUDA llama.cpp Gemma 4 QAT VLM backend
  server: ${LLAMA_SERVER_BIN}
  model: ${MODEL_FILE}
  mmproj: ${MMPROJ_FILE}
  alias: ${ALIAS}
  listen: http://${HOST}:${PORT}
  ctx: ${CTX_SIZE}
  gpu layers: ${N_GPU_LAYERS}

This follows Unsloth's Gemma 4 QAT llama.cpp guide:
  model repo: unsloth/gemma-4-26B-A4B-it-qat-GGUF
  quant: UD-Q4_K_XL
  mmproj: mmproj-BF16.gguf
EOF

exec "$LLAMA_SERVER_BIN"   --model "$MODEL_FILE"   --mmproj "$MMPROJ_FILE"   --host "$HOST"   --port "$PORT"   --ctx-size "$CTX_SIZE"   --n-gpu-layers "$N_GPU_LAYERS"   --temp 1.0   --top-p 0.95   --top-k 64   --alias "$ALIAS"   --chat-template-kwargs '{"enable_thinking":false}'   ${LLAMA_EXTRA_ARGS:-}
