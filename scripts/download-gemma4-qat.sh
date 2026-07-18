#!/usr/bin/env bash
set -euo pipefail

CANDIDATE="${CANDIDATE:-31b}"
MMPROJ_FILE="mmproj-BF16.gguf"

case "$CANDIDATE" in
  31b)
    MODEL_REPO="unsloth/gemma-4-31B-it-qat-GGUF"
    REVISION="43cc1aeb31adf47ec06a854507ce552cd9862e6f"
    SERVED_ALIAS="unsloth/gemma-4-31B-it-qat-GGUF-schema-43cc1aeb"
    MODEL_FILE="gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
    MTP_FILE="mtp-gemma-4-31B-it.gguf"
    MODEL_BYTES=17287670048
    MMPROJ_BYTES=1200726496
    MTP_BYTES=279955968
    MODEL_SHA256="00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71"
    MMPROJ_SHA256="d904b3579a9fbfbd50bc9bf40cb7384909edbd69fa9276db5ddc853e80f0edca"
    MTP_SHA256="3a5e99fd8d0b23afb1fccd1ee0c9ebd1f571d00399c2dae2292d217feeec0f6b"
    ;;
  26b)
    MODEL_REPO="unsloth/gemma-4-26B-A4B-it-qat-GGUF"
    REVISION="7b92b5b28818151e8669af2e45e88d6086f490dd"
    SERVED_ALIAS="unsloth/gemma-4-26B-A4B-it-qat-GGUF-schema-7b92b5b2"
    MODEL_FILE="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
    MTP_FILE="mtp-gemma-4-26B-A4B-it.gguf"
    MODEL_BYTES=14249047104
    MMPROJ_BYTES=1194828256
    MTP_BYTES=251939328
    MODEL_SHA256="a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891"
    MMPROJ_SHA256="7b06953ccdbe8cf363f47841a7afaacd2b1c2ff9a8d6b426fdec7521a6878744"
    MTP_SHA256="7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a"
    ;;
  *)
    echo "CANDIDATE must be 31b or 26b" >&2
    exit 2
    ;;
esac

MODEL_ROOT="${MODEL_ROOT:-$HOME/models/gemma4-schema}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/${CANDIDATE}-${REVISION}}"
STAGE_DIR="$MODEL_DIR/.hf-stage"
RESERVE_BYTES=$((10 * 1024 * 1024 * 1024))
LLAMA_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:8d1e8ddc42585632d7bd625c2285eba891ed2ed4428e9eda25ca71ce2f6cce27"
LLAMA_BUILD="b9967 (4f37f519722aa3242eecb7649466b4a4a2d6d6da)"

command -v hf >/dev/null || {
  echo "Install Hugging Face Hub first: python3 -m pip install --user --upgrade huggingface_hub hf_xet" >&2
  exit 1
}
command -v jq >/dev/null || {
  echo "jq is required to create the canonical model lock" >&2
  exit 1
}
hf auth whoami >/dev/null
mkdir -p "$MODEL_DIR" "$STAGE_DIR"

verify_artifact() {
  local path="$1" expected_bytes="$2" expected_sha="$3"
  [[ -f "$path" ]] || return 1
  [[ "$(stat -c %s "$path")" == "$expected_bytes" ]] || return 1
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected_sha" ]]
}

FILES=("$MODEL_FILE" "$MMPROJ_FILE" "$MTP_FILE")
BYTES=("$MODEL_BYTES" "$MMPROJ_BYTES" "$MTP_BYTES")
HASHES=("$MODEL_SHA256" "$MMPROJ_SHA256" "$MTP_SHA256")
PENDING=()
REQUIRED_BYTES=0

for index in "${!FILES[@]}"; do
  if ! verify_artifact "$MODEL_DIR/${FILES[$index]}" "${BYTES[$index]}" "${HASHES[$index]}"; then
    PENDING+=("${FILES[$index]}")
    REQUIRED_BYTES=$((REQUIRED_BYTES + BYTES[index]))
  fi
done

if ((${#PENDING[@]})); then
  AVAILABLE_BYTES="$(df --output=avail -B1 "$MODEL_DIR" | tail -n 1 | tr -d ' ')"
  if ((AVAILABLE_BYTES - REQUIRED_BYTES < RESERVE_BYTES)); then
    echo "Insufficient space: need $REQUIRED_BYTES bytes plus a $RESERVE_BYTES byte reserve; have $AVAILABLE_BYTES" >&2
    exit 1
  fi

  HF_XET_HIGH_PERFORMANCE=1 hf download "$MODEL_REPO" "${PENDING[@]}" \
    --revision "$REVISION" --local-dir "$STAGE_DIR" --max-workers 3
fi

for index in "${!FILES[@]}"; do
  file="${FILES[$index]}"
  if verify_artifact "$MODEL_DIR/$file" "${BYTES[$index]}" "${HASHES[$index]}"; then
    continue
  fi
  verify_artifact "$STAGE_DIR/$file" "${BYTES[$index]}" "${HASHES[$index]}" || {
    echo "Downloaded artifact failed size/hash verification: $file" >&2
    exit 1
  }
  mv -f "$STAGE_DIR/$file" "$MODEL_DIR/$file.part"
  mv -f "$MODEL_DIR/$file.part" "$MODEL_DIR/$file"
done

LOCK_PAYLOAD="$(jq -cS -n \
  --arg source "$MODEL_REPO" --arg revision "$REVISION" \
  --arg alias "$SERVED_ALIAS" --arg llama_image "$LLAMA_IMAGE" --arg llama_build "$LLAMA_BUILD" \
  --arg model_path "$MODEL_FILE" --arg model_sha "$MODEL_SHA256" --argjson model_bytes "$MODEL_BYTES" \
  --arg mmproj_path "$MMPROJ_FILE" --arg mmproj_sha "$MMPROJ_SHA256" --argjson mmproj_bytes "$MMPROJ_BYTES" \
  --arg mtp_path "$MTP_FILE" --arg mtp_sha "$MTP_SHA256" --argjson mtp_bytes "$MTP_BYTES" \
  '{schema:1,source:$source,revision:$revision,served_alias:$alias,SCHEMA_LOCAL_MODEL:$alias,
    llama_cpp:{image:$llama_image,build:$llama_build},
    artifacts:[
      {path:$model_path,bytes:$model_bytes,sha256:$model_sha},
      {path:$mmproj_path,bytes:$mmproj_bytes,sha256:$mmproj_sha},
      {path:$mtp_path,bytes:$mtp_bytes,sha256:$mtp_sha}
    ]}')"
LOCK_SHA256="$(printf '%s' "$LOCK_PAYLOAD" | sha256sum | cut -d ' ' -f 1)"
jq -cS --arg lock_sha256 "$LOCK_SHA256" '. + {lock_sha256:$lock_sha256}' \
  <<<"$LOCK_PAYLOAD" >"$MODEL_DIR/model-lock.json.part"
mv -f "$MODEL_DIR/model-lock.json.part" "$MODEL_DIR/model-lock.json"

echo "Downloaded, verified, and locked $MODEL_REPO@$REVISION in $MODEL_DIR"
