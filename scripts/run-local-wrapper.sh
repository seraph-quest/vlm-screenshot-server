#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install -r requirements.txt

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export VLM_BASE_URL="${VLM_BASE_URL:-http://192.168.1.26:8000/v1}"
export VLM_MODEL="${VLM_MODEL:-unsloth/gemma-4-31B-it-qat-GGUF-schema-43cc1aeb}"
export VLM_TIMEOUT_SECONDS="${VLM_TIMEOUT_SECONDS:-180}"
export VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-700}"
export VLM_TEMPERATURE="${VLM_TEMPERATURE:-0}"
export REDACT_VISIBLE_TEXT="${REDACT_VISIBLE_TEXT:-true}"

echo "Starting VLM screenshot wrapper on http://${HOST}:${PORT}"
echo "Forwarding requests to ${VLM_BASE_URL} (${VLM_MODEL})"
exec "$VENV_DIR/bin/uvicorn" app.main:app --host "$HOST" --port "$PORT"
