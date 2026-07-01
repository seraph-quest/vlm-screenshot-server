# VLM Screenshot Server

Dockerized LAN/local screenshot analysis service for Seraph, Framekeeper output folders, or any app that wants to send screenshots to a private vision-language model.

The service does not take screenshots. It only accepts image bytes and forwards them to an OpenAI-compatible VLM endpoint such as vLLM, SGLang, llama.cpp server, LM Studio, or an Ollama-compatible proxy.

## Target Setup

- Seraph runs on your Mac/Linux workstation.
- The GPU model server runs on another machine with an RTX 3090 Ti 24 GB.
- This repo runs a small HTTP API near Seraph or near the GPU host and forwards screenshots to the configured VLM backend.

```mermaid
flowchart LR
  A["Screenshot folder"] --> B["Seraph"]
  B --> C["VLM Screenshot Server"]
  C --> D["GPU host: vLLM/SGLang/llama.cpp"]
```

## Quick Start

### Recommended Seraph Topology

Run the GPU model server on the RTX 3090 Ti host and run this wrapper natively on the Mac where Seraph runs. This avoids Docker Desktop networking problems with private LAN GPU hosts.

```mermaid
flowchart LR
  A["Seraph on Mac"] --> B["Wrapper on Mac: 127.0.0.1:8000"]
  B --> C["GPU llama server: 192.168.1.26:8000/v1"]
```

On the GPU host, use the official Unsloth Gemma 4 QAT llama.cpp path. Build a CUDA llama.cpp, download the QAT GGUF and multimodal projector, then start the server:

```bash
git clone https://github.com/seraph-quest/vlm-screenshot-server.git
cd vlm-screenshot-server
./scripts/build-llama-cuda.sh
./scripts/download-gemma4-qat.sh
./scripts/run-gemma4-3090ti.sh
```

Equivalent explicit server command after the build/download steps:

```bash
$HOME/src/llama.cpp/llama-server \
  --model $HOME/models/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj $HOME/models/gemma-4-26B-A4B-it-qat-GGUF/mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --ctx-size 8192 \
  --n-gpu-layers 999 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --alias unsloth/gemma-4-26B-A4B-it-qat-GGUF \
  --chat-template-kwargs '{"enable_thinking":false}'
```

This intentionally does not use `llama serve -hf ... --no-mmproj-offload`; that path made text generation fast but left screenshot vision requests around 75 seconds on the RTX 3090 Ti. The QAT path uses Unsloth's `UD-Q4_K_XL` model plus explicit `mmproj-BF16.gguf` with a CUDA-built `llama-server`.

On the Mac running Seraph:

```bash
cp .env.example .env
./scripts/run-local-wrapper.sh
```

Expected `.env` values for this topology:

```env
HOST=127.0.0.1
PORT=8000
VLM_BASE_URL=http://192.168.1.26:8000/v1
VLM_MODEL=unsloth/gemma-4-26B-A4B-it-qat-GGUF
VLM_TRUST_ENV=false
```

Health checks from the Mac:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/backend
```

Analyze a screenshot:

```bash
curl -F "file=@/path/to/screenshot.png" \
  http://127.0.0.1:8000/v1/analyze-file
```

Seraph should point at the local wrapper, not the GPU host directly:

```env
SERAPH_SCREEN_ANALYSIS_PROVIDER=local-vlm
SERAPH_LOCAL_VLM_BASE_URL=http://127.0.0.1:8000
SERAPH_LOCAL_VLM_MODEL=unsloth/gemma-4-26B-A4B-it-qat-GGUF
```

### Docker Wrapper

Docker is still supported when the container can reach the GPU backend. It is not the recommended Mac-to-LAN default because Docker Desktop networking may not reach some private GPU host bindings.

```bash
docker compose up --build
```

For gated model repos, set `HUGGING_FACE_HUB_TOKEN` in `.env` after accepting the model license.

## GPU Backend Examples

### vLLM

```bash
vllm serve google/gemma-4-12b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

### llama.cpp Server

For Gemma 4 QAT on RTX 3090 Ti, use the CUDA llama.cpp build and QAT runner from this repo:

```bash
./scripts/build-llama-cuda.sh
./scripts/download-gemma4-qat.sh
./scripts/run-gemma4-3090ti.sh
```

Set the wrapper to call it:

```env
VLM_BASE_URL=http://GPU_SERVER_IP:8000/v1
```

## Model Shortlist For RTX 3090 Ti 24 GB

This list is oriented toward screenshot analysis: OCR, UI understanding, layout, code/editor text, terminal output, browser pages, dashboards, and concise JSON extraction.

| Rank | Model | Fit posture | Why use it | Notes |
| --- | --- | --- | --- | --- |
| 1 | Gemma 4 26B-A4B Dynamic 4-bit / GGUF via Unsloth | Strong 24 GB candidate | Best Gemma-first target if the quantized runtime fits and quality is stable | Unsloth lists 26B-A4B GGUF 4-bit around 16 GB total memory and Dynamic 4-bit around 19.78 GB. |
| 2 | Gemma 4 12B, preferably QAT/Q4 when available in your runtime | Expected sweet spot | Strong Google multimodal model family, QAT/Q4 options, likely good quality/speed balance on 24 GB | Start here if 26B-A4B is too slow or unavailable. Use the exact current model ID from Google/Hugging Face/Kaggle. |
| 3 | MiniCPM-V 4.5 AWQ/GPTQ/GGUF | Strong practical fit | Excellent small VLM option with OCR/document focus and quantized releases | Best non-Gemma first comparison. |
| 4 | Qwen2.5-VL-32B-Instruct-AWQ | Tight but high quality | Strong OCR, UI/chart/document reasoning; official AWQ quantization | Try with lower context/resolution/concurrency. |
| 5 | GLM-4.1V-9B-Thinking | Practical fit | Good reasoning-style VLM around screenshot interpretation | May be slower/more verbose than needed. |
| 6 | InternVL3-14B-AWQ | Practical/tight fit | Strong open VLM family with quantized options | Good benchmark candidate if Qwen32 is too heavy. |
| Watch | Qwen3 3.6B / "Qwen 3.6" | Verify before using | Could be useful if a vision-capable checkpoint exists, but plain Qwen3 text models do not replace a VLM | Do not use for screenshot analysis unless the model card explicitly supports image inputs. |

## Current Benchmark Notes

Use these as research anchors, not marketing gospel. For Seraph, benchmark on your own screenshot corpus because UI/OCR quality is workload-specific.

- Google lists Gemma 4 as multimodal, with E2B/E4B efficiency models and 12B/26B/31B advanced reasoning models, and positions the family for cloud servers, laptops, phones, and personal computers.
- Google’s Gemma 4 QAT release notes describe quantization-aware trained checkpoints, Q4_0 artifacts, and runtime support across vLLM, SGLang, llama.cpp, Ollama, and LM Studio.
- Unsloth’s Gemma 4 page is important for 24 GB cards because it lists practical 4-bit/Dynamic 4-bit memory footprints, including Gemma 4 26B-A4B around the high-teens GB range.
- MiniCPM-V 4.5 is a strong small-model baseline for OCR/document-style work and has quantized variants.
- Qwen2.5-VL-32B-AWQ is the quality-stress test for a 24 GB card: likely better on hard screenshots, but more memory-sensitive.
- Qwen3/Qwen 3.6 should stay on the watchlist until the exact candidate is confirmed as a vision-language model. Text-only Qwen3 checkpoints are not suitable for screenshot analysis.

Sources checked June 30, 2026:

- [Gemma model page](https://deepmind.google/models/gemma/)
- [Gemma 4 QAT announcement](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- [Unsloth Gemma 4 models](https://unsloth.ai/docs/models/gemma-4)
- [MiniCPM-V 4.5 model card](https://huggingface.co/openbmb/MiniCPM-V-4_5)
- [Qwen2.5-VL-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ)
- [GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
- [InternVL3-14B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-14B-AWQ)

## Seraph Integration Shape

Seraph should call this service as a remote image analyzer:

```env
SERAPH_SCREEN_ANALYSIS_PROVIDER=local-vlm
SERAPH_LOCAL_VLM_BASE_URL=http://127.0.0.1:8000
SERAPH_LOCAL_VLM_MODEL=unsloth/gemma-4-26B-A4B-it-qat-GGUF
```

This repo intentionally keeps the screenshot producer separate from analysis: screenshots are just files or uploaded image bytes.

## API

### `POST /v1/analyze-file`

Multipart upload:

```bash
curl -F "file=@screen.png" http://127.0.0.1:8000/v1/analyze-file
```

Seraph may also send profile-control form fields:

```bash
curl \
  -F "file=@screen.png" \
  -F "runtime_profile=screenshot_fast" \
  -F "runtime_path=screenshot_image_analysis" \
  -F "priority=background" \
  -F "reasoning=off" \
  -F 'profile_options={"chat_template_kwargs":{"enable_thinking":false},"reasoning":false,"reasoning_format":"none"}' \
  http://127.0.0.1:8000/v1/analyze-file
```

For `runtime_profile=screenshot_fast` or `reasoning=off`, the wrapper normalizes Gemma channel markers such as `<|channel>thought` before returning/parsing the response. This proves that callers do not receive visible reasoning markers after gateway normalization; it does not prove the backend performed no internal reasoning.

### Queue and Backpressure

All backend model calls pass through a bounded priority queue before hitting the OpenAI-compatible GPU server. Default behavior favors interactive/chat traffic over background screenshot analysis:

```env
QUEUE_MAX_SIZE=8
QUEUE_WORKERS=2
QUEUE_BACKGROUND_WORKERS=1
QUEUE_ADMIT_TIMEOUT_SECONDS=1
QUEUE_RESULT_TIMEOUT_SECONDS=600
```

Priority order is `interactive`, `high`, `normal`, `background`, then `low`. By default, one worker is reserved away from background screenshot work, so a running screenshot does not consume all wrapper capacity. The effective background worker count is clamped below total workers; for example, `QUEUE_WORKERS=1` leaves no background slot because otherwise screenshot work could starve chat/report traffic. If the queue is full of lower-priority work, a higher-priority request can preempt queued lower-priority work instead of being rejected. This keeps Seraph chat/report requests from being starved by a large screenshot backlog. If no lower-priority item can be preempted before `QUEUE_ADMIT_TIMEOUT_SECONDS`, the wrapper returns HTTP 429 with queue status. If a queued request waits/runs longer than `QUEUE_RESULT_TIMEOUT_SECONDS`, it returns HTTP 504.

### `POST /v1/chat/completions`

OpenAI-compatible chat forwarding is disabled by default. Enable it only for Seraph local LLM routing on localhost or a private network:

```env
CHAT_PROXY_ENABLED=true
CHAT_PROXY_API_KEY=use-a-local-secret
```

Seraph must send the same value as `LOCAL_LLM_API_KEY`. The same screenshot-fast marker normalization is applied when request metadata or `X-Seraph-*` headers identify the request as `screenshot_fast` / `reasoning=off`.

### `POST /v1/analyze`

JSON body:

```json
{
  "image_base64": "iVBORw0KGgo...",
  "media_type": "image/png",
  "app_hint": "VS Code"
}
```

Response:

```json
{
  "provider": "openai-compatible-vlm",
  "model": "google/gemma-4-12b-it",
  "duration_ms": 1234,
  "analysis": {
    "activity": "coding",
    "app_guess": "VS Code",
    "project": "seraph",
    "summary": "Editing a screenshot analysis provider.",
    "visible_text": ["redacted or short snippets"],
    "sensitive": false,
    "confidence": 0.86
  },
  "raw_text": "{...}"
}
```

## Privacy

- This service does not persist images.
- Raw screenshots are forwarded to the configured VLM backend.
- Put the VLM backend on a private LAN/VPN.
- Keep `REDACT_VISIBLE_TEXT=true` unless you intentionally want raw snippets returned.
- Do not expose this service or the VLM backend directly to the public internet.
