from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
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
    app_hint: Optional[str] = None,
    prompt: Optional[str] = None,
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
    )


async def _analyze_image(
    image_bytes: bytes,
    *,
    media_type: str,
    app_hint: Optional[str],
    prompt: Optional[str],
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
