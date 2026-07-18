#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_CPP_DIR/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma4-schema/31b-43cc1aeb31adf47ec06a854507ce552cd9862e6f}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-$MODEL_DIR/mmproj-BF16.gguf}"
MTP_DRAFTER_FILE="${MTP_DRAFTER_FILE:-$MODEL_DIR/mtp-gemma-4-31B-it.gguf}"
ALIAS="${ALIAS:-unsloth/gemma-4-31B-it-qat-GGUF-schema-43cc1aeb}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CTX_SIZE="${CTX_SIZE:-32768}"
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

EXTRA_ARGS=()
if [ "${ENABLE_MTP:-0}" = "1" ]; then
  if [ ! -f "$MTP_DRAFTER_FILE" ]; then
    echo "Missing matching MTP drafter: $MTP_DRAFTER_FILE" >&2
    exit 1
  fi
  EXTRA_ARGS=(--spec-type draft-mtp --spec-draft-model "$MTP_DRAFTER_FILE" --spec-draft-n-max 4 --flash-attn on)
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
  model repo: unsloth/gemma-4-31B-it-qat-GGUF
  quant: UD-Q4_K_XL
  mmproj: mmproj-BF16.gguf
EOF

exec "$LLAMA_SERVER_BIN" \
  --model "$MODEL_FILE" --mmproj "$MMPROJ_FILE" \
  --host "$HOST" --port "$PORT" --ctx-size "$CTX_SIZE" \
  --n-gpu-layers "$N_GPU_LAYERS" --temp 1.0 --top-p 0.95 --top-k 64 \
  --alias "$ALIAS" --chat-template-kwargs '{"enable_thinking":false}' \
  "${EXTRA_ARGS[@]}" ${LLAMA_EXTRA_ARGS:-}
