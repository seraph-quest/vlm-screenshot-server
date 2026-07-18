from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIGEST = "8d1e8ddc42585632d7bd625c2285eba891ed2ed4428e9eda25ca71ce2f6cce27"
REVISION_31B = "43cc1aeb31adf47ec06a854507ce552cd9862e6f"
REVISION_26B = "7b92b5b28818151e8669af2e45e88d6086f490dd"


def test_gpu_compose_uses_pinned_llama_cpp_and_refreshed_31b_defaults() -> None:
    compose = (ROOT / "docker-compose.gpu.yml").read_text()
    assert f"server-cuda@sha256:{LLAMA_DIGEST}" in compose
    assert "vllm/vllm-openai:latest" not in compose
    assert "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf" in compose
    assert "unsloth/gemma-4-31B-it-qat-GGUF-schema-43cc1aeb" in compose
    assert "--mmproj" in compose and ":/models:ro" in compose
    assert "condition: service_healthy" in compose


def test_mtp_is_opt_in_and_uses_matching_drafter_contract() -> None:
    base = (ROOT / "docker-compose.gpu.yml").read_text()
    mtp = (ROOT / "docker-compose.gpu-mtp.yml").read_text()
    assert "--spec-type" not in base
    for value in ("--spec-type", "draft-mtp", "--spec-draft-model", "--spec-draft-n-max", '"4"', "--flash-attn", "on"):
        assert value in mtp
    assert "mtp-gemma-4-31B-it.gguf" in mtp


def test_downloader_pins_both_official_revisions_and_hashes() -> None:
    script = (ROOT / "scripts/download-gemma4-qat.sh").read_text()
    assert REVISION_31B in script and REVISION_26B in script
    for digest in (
        "00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71",
        "d904b3579a9fbfbd50bc9bf40cb7384909edbd69fa9276db5ddc853e80f0edca",
        "3a5e99fd8d0b23afb1fccd1ee0c9ebd1f571d00399c2dae2292d217feeec0f6b",
        "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891",
        "7b06953ccdbe8cf363f47841a7afaacd2b1c2ff9a8d6b426fdec7521a6878744",
        "7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a",
    ):
        assert digest in script
    assert "HF_XET_HIGH_PERFORMANCE=1" in script
    for size in ("17287670048", "1200726496", "279955968", "14249047104", "1194828256", "251939328"):
        assert size in script
    assert "RESERVE_BYTES=$((10 * 1024 * 1024 * 1024))" in script
    assert 'STAGE_DIR="$MODEL_DIR/.hf-stage"' in script
    assert '"$MODEL_DIR/$file.part"' in script
    assert "model-lock.json.part" in script and "lock_sha256" in script
    assert "SCHEMA_LOCAL_MODEL" in script and "LLAMA_IMAGE" in script and "LLAMA_BUILD" in script
