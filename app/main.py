from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.config import settings


app = FastAPI(
    title="VLM Screenshot Server",
    version="0.1.0",
    description="Local/LAN screenshot analysis API backed by an OpenAI-compatible vision model.",
)


class AnalyzeImageUrlRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded PNG/JPEG image bytes.")
    media_type: str = "image/png"
    app_hint: Optional[str] = None
    prompt: Optional[str] = None


class AnalyzeResponse(BaseModel):
    provider: str
    model: str
    duration_ms: int
    analysis: dict[str, Any]
    raw_text: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "backend": settings.vlm_base_url,
        "model": settings.vlm_model,
    }


@app.get("/health/backend")
async def backend_health() -> dict[str, Any]:
    """Check whether the configured OpenAI-compatible VLM backend is reachable."""
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=settings.vlm_trust_env) as client:
            response = await client.get(_backend_health_url(settings.vlm_base_url))
            if response.status_code == 404:
                response = await client.get(_models_url(settings.vlm_base_url))
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"VLM backend health HTTP {exc.response.status_code}: {exc.response.text[:800]}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"VLM backend health unavailable: {exc}") from exc
    return {
        "status": "ok",
        "backend": settings.vlm_base_url,
        "model": settings.vlm_model,
        "backend_status": response.status_code,
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_base64(body: AnalyzeImageUrlRequest) -> AnalyzeResponse:
    image_bytes = _decode_base64(body.image_base64)
    return await _analyze_image(
        image_bytes,
        media_type=body.media_type,
        app_hint=body.app_hint,
        prompt=body.prompt,
    )


@app.post("/v1/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(
    file: UploadFile = File(...),
    app_hint: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    runtime_profile: Optional[str] = Form(default=None),
    runtime_path: Optional[str] = Form(default=None),
    priority: Optional[str] = Form(default=None),
    reasoning: Optional[str] = Form(default=None),
    profile_options: Optional[str] = Form(default=None),
) -> AnalyzeResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image upload")
    media_type = file.content_type or _guess_media_type(file.filename or "")
    return await _analyze_image(
        image_bytes,
        media_type=media_type,
        app_hint=app_hint,
        prompt=prompt,
        runtime_profile=runtime_profile,
        runtime_path=runtime_path,
        priority=priority,
        reasoning=reasoning,
        profile_options=profile_options,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    """Forward OpenAI-compatible text/chat requests to the configured backend."""
    if not settings.chat_proxy_enabled:
        raise HTTPException(status_code=404, detail="chat proxy is disabled")
    _require_chat_proxy_auth(request)
    payload = await request.json()
    runtime_profile = _runtime_profile_from_request(payload, request)
    reasoning = _reasoning_from_request(payload, request)
    payload.setdefault("model", settings.vlm_model)
    headers = {"Content-Type": "application/json"}
    if settings.vlm_api_key:
        headers["Authorization"] = f"Bearer {settings.vlm_api_key}"

    try:
        async with httpx.AsyncClient(timeout=settings.vlm_timeout_seconds, trust_env=settings.vlm_trust_env) as client:
            response = await client.post(
                _chat_completions_url(settings.vlm_base_url),
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800]
        raise HTTPException(status_code=502, detail=f"VLM backend HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"VLM backend unavailable: {exc}") from exc
    response_payload = response.json()
    if _should_sanitize_visible_reasoning(runtime_profile=runtime_profile, reasoning=reasoning):
        _sanitize_chat_completion_payload(response_payload)
    return response_payload


def _require_chat_proxy_auth(request: Request) -> None:
    configured_key = settings.chat_proxy_api_key.strip()
    if not configured_key:
        raise HTTPException(status_code=403, detail="chat proxy auth is not configured")
    expected = f"Bearer {configured_key}"
    if request.headers.get("Authorization") != expected:
        raise HTTPException(status_code=401, detail="chat proxy auth required")


async def _analyze_image(
    image_bytes: bytes,
    *,
    media_type: str,
    app_hint: Optional[str],
    prompt: Optional[str],
    runtime_profile: Optional[str] = None,
    runtime_path: Optional[str] = None,
    priority: Optional[str] = None,
    reasoning: Optional[str] = None,
    profile_options: Optional[str] = None,
) -> AnalyzeResponse:
    start = time.monotonic()
    image_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    payload = {
        "model": settings.vlm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or _default_prompt(app_hint)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": settings.vlm_max_tokens,
        "temperature": settings.vlm_temperature,
    }
    profile_payload_options = _profile_options_dict(profile_options)
    payload.update(profile_payload_options)
    if runtime_profile or runtime_path or priority or reasoning:
        metadata = dict(payload.get("metadata") or {})
        if runtime_profile:
            metadata["runtime_profile"] = runtime_profile
        if runtime_path:
            metadata["runtime_path"] = runtime_path
        if priority:
            metadata["priority"] = priority
        if reasoning:
            metadata["reasoning"] = reasoning
        payload["metadata"] = metadata
    headers = {"Content-Type": "application/json"}
    if settings.vlm_api_key:
        headers["Authorization"] = f"Bearer {settings.vlm_api_key}"

    try:
        async with httpx.AsyncClient(timeout=settings.vlm_timeout_seconds, trust_env=settings.vlm_trust_env) as client:
            response = await client.post(
                _chat_completions_url(settings.vlm_base_url),
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800]
        raise HTTPException(status_code=502, detail=f"VLM backend HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"VLM backend unavailable: {exc}") from exc

    raw_text = response.json()["choices"][0]["message"]["content"].strip()
    if _should_sanitize_visible_reasoning(runtime_profile=runtime_profile, reasoning=reasoning):
        raw_text = _strip_visible_reasoning(raw_text).strip()
    parsed = _parse_json_object(raw_text)
    if settings.redact_visible_text and isinstance(parsed.get("visible_text"), list):
        parsed["visible_text"] = [_redact_sensitive_text(str(item)) for item in parsed["visible_text"][:20]]
    duration_ms = int((time.monotonic() - start) * 1000)
    return AnalyzeResponse(
        provider="openai-compatible-vlm",
        model=settings.vlm_model,
        duration_ms=duration_ms,
        analysis=parsed,
        raw_text=raw_text,
    )


def _default_prompt(app_hint: Optional[str]) -> str:
    app_line = f"Current app hint: {app_hint}\n" if app_hint else ""
    return (
        "Analyze this desktop screenshot for productivity/activity reporting.\n"
        f"{app_line}"
        "Return only valid JSON with this shape:\n"
        "{\n"
        '  "activity": "coding|reading|writing|browser|chat|design|terminal|meeting|unknown",\n'
        '  "app_guess": "short app or website guess",\n'
        '  "project": "project/workstream name or null",\n'
        '  "summary": "one concise sentence",\n'
        '  "visible_text": ["short non-sensitive snippets only"],\n'
        '  "sensitive": true,\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Do not include secrets, tokens, passwords, credit card numbers, private messages, or long verbatim text."
    )


def _chat_completions_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def _models_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean.rsplit("/chat/completions", 1)[0] + "/models"
    if clean.endswith("/v1"):
        return f"{clean}/models"
    return f"{clean}/v1/models"


def _backend_health_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/v1"):
        return clean[:-3] + "/health"
    if clean.endswith("/chat/completions"):
        return clean.rsplit("/v1/chat/completions", 1)[0] + "/health"
    return f"{clean}/health"


def _decode_base64(value: str) -> bytes:
    try:
        if "," in value and value.strip().startswith("data:"):
            value = value.split(",", 1)[1]
        decoded = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid base64 image") from exc
    if not decoded:
        raise HTTPException(status_code=400, detail="empty image")
    return decoded


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"VLM returned non-JSON output: {raw_text[:500]}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=502, detail="VLM JSON output must be an object")
    return parsed


def _profile_options_dict(raw_options: Optional[str]) -> dict[str, Any]:
    if not raw_options:
        return {}
    try:
        parsed = json.loads(raw_options)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _runtime_profile_from_request(payload: dict[str, Any], request: Request) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return str(
        metadata.get("runtime_profile")
        or payload.get("runtime_profile")
        or request.headers.get("X-Seraph-Runtime-Profile")
        or ""
    ).strip()


def _reasoning_from_request(payload: dict[str, Any], request: Request) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    value = (
        metadata.get("reasoning")
        if "reasoning" in metadata
        else payload.get("reasoning", request.headers.get("X-Seraph-Reasoning", ""))
    )
    return str(value).strip().lower()


def _should_sanitize_visible_reasoning(*, runtime_profile: Optional[str], reasoning: Optional[str]) -> bool:
    normalized_profile = (runtime_profile or "").strip().lower().replace("-", "_")
    normalized_reasoning = (reasoning or "").strip().lower()
    return normalized_profile == "screenshot_fast" or normalized_reasoning in {"0", "false", "no", "off"}


def _sanitize_chat_completion_payload(payload: dict[str, Any]) -> None:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _strip_visible_reasoning(content).strip()


def _strip_visible_reasoning(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    channel_match = re.search(r"(?is)<\|channel\>\s*final\s*<\|message\>(.*)", text)
    if channel_match:
        return channel_match.group(1).strip()
    thought_match = re.search(r"(?is)<\|channel\>\s*thought\s*<\|message\>.*?(<\|channel\>\s*final\s*<\|message\>.*)", text)
    if thought_match:
        return _strip_visible_reasoning(thought_match.group(1))
    mislabeled_content_match = re.fullmatch(r"(?is)<\|channel\>\s*thought\s*<channel\|>\s*(.*)", text)
    if mislabeled_content_match:
        return mislabeled_content_match.group(1).strip()
    mislabeled_message_match = re.fullmatch(r"(?is)<\|channel\>\s*thought\s*<\|message\>\s*(.*)", text)
    if mislabeled_message_match:
        return mislabeled_message_match.group(1).strip()
    text = re.sub(r"(?is)<\|channel\>\s*thought\s*<\|message\>.*?(?=<\|channel\>|$)", "", text)
    text = re.sub(r"(?is)<\|channel\>\s*thought\s*<channel\|>", "", text)
    text = re.sub(r"(?is)<\|start\|>assistant\s*<\|channel\>\s*final\s*<\|message\>", "", text)
    return text.strip()


def _redact_sensitive_text(value: str) -> str:
    patterns = [
        r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*\S+",
        r"\b(?:\d[ -]*?){13,19}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ]
    redacted = value
    for pattern in patterns:
        redacted = re.sub(pattern, "[redacted]", redacted)
    return redacted[:240]


def _guess_media_type(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    return "image/png"
