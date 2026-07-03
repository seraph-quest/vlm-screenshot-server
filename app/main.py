from __future__ import annotations

import base64
import asyncio
import contextlib
import heapq
import itertools
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings


app = FastAPI(
    title="VLM Screenshot Server",
    version="0.1.0",
    description="Local/LAN screenshot analysis API backed by an OpenAI-compatible vision model.",
)

_PRIORITY_RANKS = {
    "interactive": 0,
    "high": 1,
    "normal": 2,
    "background": 3,
    "low": 4,
}
_queue_counter = itertools.count()
_work_queue: "PriorityWorkQueue" | None = None
_workers: list[asyncio.Task[None]] = []


@dataclass
class QueuedWork:
    label: str
    priority: str
    submitted_at: float
    run: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]


class PriorityWorkQueue:
    def __init__(self, *, max_size: int, max_background_active: int) -> None:
        self._max_size = max(max_size, 1)
        self._max_background_active = max(max_background_active, 0)
        self._active_background = 0
        self._active = 0
        self._items: list[tuple[int, int, QueuedWork]] = []
        self._condition = asyncio.Condition()

    @property
    def max_size(self) -> int:
        return self._max_size

    def qsize(self) -> int:
        return len(self._items)

    def active_background(self) -> int:
        return self._active_background

    def active(self) -> int:
        return self._active

    async def put(self, item: tuple[int, int, QueuedWork]) -> None:
        rank, _, queued = item
        async with self._condition:
            if len(self._items) >= self._max_size and not self._admit_by_evicting_lower_priority(rank):
                await self._condition.wait_for(lambda: len(self._items) < self._max_size)
            heapq.heappush(self._items, item)
            self._condition.notify()

    async def get(self) -> tuple[int, int, QueuedWork]:
        async with self._condition:
            await self._condition.wait_for(self._has_eligible_work)
            item = self._pop_eligible_work()
            self._active += 1
            if _is_background_priority(item[2].priority):
                self._active_background += 1
            self._condition.notify()
            return item

    async def task_done(self, queued: QueuedWork) -> None:
        async with self._condition:
            if self._active > 0:
                self._active -= 1
            if _is_background_priority(queued.priority) and self._active_background > 0:
                self._active_background -= 1
            self._condition.notify_all()

    def _admit_by_evicting_lower_priority(self, incoming_rank: int) -> bool:
        if not self._items:
            return True
        worst_index, (worst_rank, _, worst_work) = max(enumerate(self._items), key=lambda entry: entry[1][0])
        if incoming_rank >= worst_rank:
            return False
        self._items.pop(worst_index)
        heapq.heapify(self._items)
        if not worst_work.future.done():
            worst_work.future.set_exception(
                HTTPException(
                    status_code=429,
                    detail={
                        "error": "vlm_queue_preempted",
                        "priority": worst_work.priority,
                    },
                )
            )
        return True

    def _has_eligible_work(self) -> bool:
        return any(self._is_eligible(item[2]) for item in self._items)

    def _pop_eligible_work(self) -> tuple[int, int, QueuedWork]:
        for index, item in enumerate(self._items):
            if self._is_eligible(item[2]):
                self._items.pop(index)
                heapq.heapify(self._items)
                return item
        raise RuntimeError("priority queue woke without eligible work")

    def _is_eligible(self, queued: QueuedWork) -> bool:
        if not _is_background_priority(queued.priority):
            return True
        if self._max_background_active <= 0:
            return False
        return self._active_background < self._max_background_active


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
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": settings.vlm_base_url,
        "model": settings.vlm_model,
        "queue": _queue_status(),
    }


@app.get("/queue/status")
async def queue_status() -> dict[str, Any]:
    return _queue_status()


@app.on_event("startup")
async def _start_queue_workers() -> None:
    global _work_queue
    if _workers:
        return
    _work_queue = PriorityWorkQueue(
        max_size=settings.queue_max_size,
        max_background_active=_effective_background_workers(),
    )
    for worker_id in range(_effective_queue_workers()):
        _workers.append(asyncio.create_task(_queue_worker(worker_id), name=f"vlm-queue-worker-{worker_id}"))


@app.on_event("shutdown")
async def _stop_queue_workers() -> None:
    global _work_queue
    for worker in _workers:
        worker.cancel()
    for worker in _workers:
        with contextlib.suppress(asyncio.CancelledError):
            await worker
    _workers.clear()
    _work_queue = None


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


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Union[dict[str, Any], StreamingResponse]:
    """Forward OpenAI-compatible text/chat requests to the configured backend."""
    if not settings.chat_proxy_enabled:
        raise HTTPException(status_code=404, detail="chat proxy is disabled")
    _require_chat_proxy_auth(request)
    payload = await request.json()
    runtime_profile = _runtime_profile_from_request(payload, request)
    reasoning = _reasoning_from_request(payload, request)
    priority = _priority_from_request(payload, request, default="interactive")
    payload.setdefault("model", settings.vlm_model)
    headers = {"Content-Type": "application/json"}
    if settings.vlm_api_key:
        headers["Authorization"] = f"Bearer {settings.vlm_api_key}"

    if _is_streaming_chat_request(payload):
        return StreamingResponse(
            _stream_queued_chat_completion(payload=payload, headers=headers, priority=priority),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    response = await _submit_backend_work(
        label="chat",
        priority=priority,
        run=lambda: _post_chat_completion(payload=payload, headers=headers),
    )
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

    raw_text = ""
    parsed: dict[str, Any] | None = None
    attempts = max(settings.vlm_analyze_attempts, 1)
    for attempt in range(1, attempts + 1):
        try:
            response = await _submit_backend_work(
                label="analyze",
                priority=priority or "background",
                run=lambda: _post_chat_completion(payload=payload, headers=headers),
            )
            raw_text = response.json()["choices"][0]["message"]["content"].strip()
            if _should_sanitize_visible_reasoning(runtime_profile=runtime_profile, reasoning=reasoning):
                raw_text = _strip_visible_reasoning(raw_text).strip()
            parsed = _parse_json_object(raw_text)
            if _should_emit_seraph_screenshot_schema(runtime_profile=runtime_profile, runtime_path=runtime_path):
                parsed = _coerce_seraph_screenshot_schema(parsed)
            break
        except HTTPException as exc:
            if exc.status_code not in {502, 503, 504} or attempt >= attempts:
                raise
            await asyncio.sleep(min(0.25 * attempt, 1.0))
    if parsed is None:
        raise HTTPException(status_code=502, detail="VLM analysis did not return JSON")
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


async def _post_chat_completion(*, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
    try:
        async with httpx.AsyncClient(timeout=settings.vlm_timeout_seconds, trust_env=settings.vlm_trust_env) as client:
            response = await client.post(
                _chat_completions_url(settings.vlm_base_url),
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800]
        raise HTTPException(status_code=502, detail=f"VLM backend HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"VLM backend unavailable: {exc}") from exc


async def _stream_queued_chat_completion(
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    priority: Optional[str],
) -> AsyncIterator[bytes]:
    queue = _ensure_queue()
    normalized_priority = _normalize_priority(priority)
    chunk_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    done: asyncio.Future[None] = loop.create_future()

    async def _run_stream() -> None:
        try:
            async with httpx.AsyncClient(timeout=settings.vlm_timeout_seconds, trust_env=settings.vlm_trust_env) as client:
                async with client.stream(
                    "POST",
                    _chat_completions_url(settings.vlm_base_url),
                    headers=headers,
                    json=payload,
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        detail = (await response.aread()).decode("utf-8", errors="replace")[:800]
                        raise HTTPException(
                            status_code=502,
                            detail=f"VLM backend HTTP {exc.response.status_code}: {detail}",
                        ) from exc
                    async for chunk in response.aiter_bytes():
                        if done.cancelled():
                            break
                        if chunk:
                            await chunk_queue.put(chunk)
        except httpx.HTTPError as exc:
            if not done.done():
                done.set_exception(HTTPException(status_code=502, detail=f"VLM backend unavailable: {exc}"))
        except Exception as exc:  # noqa: BLE001 - propagate stream failures to the response generator.
            if not done.done():
                done.set_exception(exc)
        else:
            if not done.done():
                done.set_result(None)
        finally:
            await chunk_queue.put(None)

    queued = QueuedWork(
        label="chat-stream",
        priority=normalized_priority,
        submitted_at=time.monotonic(),
        run=_run_stream,
        future=done,
    )
    item = (_priority_rank(normalized_priority), next(_queue_counter), queued)
    try:
        await asyncio.wait_for(queue.put(item), timeout=max(settings.queue_admit_timeout_seconds, 0.01))
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "vlm_queue_full",
                "priority": normalized_priority,
                "queue": _queue_status(),
            },
        ) from exc

    try:
        while True:
            chunk = await chunk_queue.get()
            if chunk is None:
                break
            yield chunk
        await done
    except asyncio.CancelledError:
        if not done.done():
            done.cancel()
        raise


async def _submit_backend_work(
    *,
    label: str,
    priority: Optional[str],
    run: Callable[[], Awaitable[Any]],
) -> Any:
    queue = _ensure_queue()
    normalized_priority = _normalize_priority(priority)
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    queued = QueuedWork(
        label=label,
        priority=normalized_priority,
        submitted_at=time.monotonic(),
        run=run,
        future=future,
    )
    item = (_priority_rank(normalized_priority), next(_queue_counter), queued)
    try:
        await asyncio.wait_for(queue.put(item), timeout=max(settings.queue_admit_timeout_seconds, 0.01))
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "vlm_queue_full",
                "priority": normalized_priority,
                "queue": _queue_status(),
            },
        ) from exc
    try:
        return await asyncio.wait_for(future, timeout=max(settings.queue_result_timeout_seconds, 1.0))
    except asyncio.TimeoutError as exc:
        if not future.done():
            future.cancel()
        raise HTTPException(
            status_code=504,
            detail={
                "error": "vlm_queue_result_timeout",
                "priority": normalized_priority,
                "queue": _queue_status(),
            },
        ) from exc


async def _queue_worker(worker_id: int) -> None:
    queue = _ensure_queue()
    while True:
        _, _, queued = await queue.get()
        try:
            if queued.future.cancelled():
                continue
            try:
                result = await queued.run()
            except Exception as exc:  # noqa: BLE001 - propagate provider errors to the caller future.
                if not queued.future.done():
                    queued.future.set_exception(exc)
            else:
                if not queued.future.done():
                    queued.future.set_result(result)
        finally:
            await queue.task_done(queued)


def _ensure_queue() -> PriorityWorkQueue:
    global _work_queue
    if _work_queue is None:
        _work_queue = PriorityWorkQueue(
            max_size=settings.queue_max_size,
            max_background_active=_effective_background_workers(),
        )
    return _work_queue


def _queue_status() -> dict[str, Any]:
    queue = _ensure_queue()
    return {
        "queued": queue.qsize(),
        "active": queue.active(),
        "active_background": queue.active_background(),
        "max_size": queue.max_size,
        "workers": _effective_queue_workers(),
        "background_workers": _effective_background_workers(),
        "configured_workers": settings.queue_workers,
        "configured_background_workers": settings.queue_background_workers,
        "admit_timeout_seconds": settings.queue_admit_timeout_seconds,
        "result_timeout_seconds": settings.queue_result_timeout_seconds,
    }


def _priority_from_request(payload: dict[str, Any], request: Request, *, default: str) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    value = metadata.get("priority") or payload.get("priority") or request.headers.get("X-Seraph-Priority") or default
    return _normalize_priority(str(value))


def _normalize_priority(priority: Optional[str]) -> str:
    normalized = (priority or "normal").strip().lower()
    return normalized if normalized in _PRIORITY_RANKS else "normal"


def _priority_rank(priority: str) -> int:
    return _PRIORITY_RANKS.get(_normalize_priority(priority), _PRIORITY_RANKS["normal"])


def _is_streaming_chat_request(payload: dict[str, Any]) -> bool:
    return payload.get("stream") is True


def _is_background_priority(priority: str) -> bool:
    return _normalize_priority(priority) in {"background", "low"}


def _effective_queue_workers() -> int:
    return max(settings.queue_workers, 1)


def _effective_background_workers() -> int:
    workers = _effective_queue_workers()
    if workers <= 1:
        return 1 if settings.queue_background_workers > 0 else 0
    return max(0, min(settings.queue_background_workers, workers - 1))


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


def _should_emit_seraph_screenshot_schema(*, runtime_profile: Optional[str], runtime_path: Optional[str]) -> bool:
    return (runtime_profile or "").strip().lower() == "screenshot_fast" or (
        runtime_path or ""
    ).strip().lower() == "screenshot_image_analysis"


def _coerce_seraph_screenshot_schema(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") == "seraph.screenshot_analysis.v1":
        normalized = dict(payload)
        for key in (
            "detailed_observations",
            "applications",
            "visible_artifacts",
            "key_visible_text",
            "privacy_notes",
            "report_tags",
        ):
            normalized[key] = _coerce_string_list(normalized.get(key))
        goal_alignment = normalized.get("goal_alignment")
        if isinstance(goal_alignment, dict):
            normalized["goal_alignment"] = {
                **goal_alignment,
                "goal_refs": _coerce_string_list(goal_alignment.get("goal_refs")),
                "evidence": _coerce_string_list(goal_alignment.get("evidence")),
            }
        return normalized
    activity = str(payload.get("activity_type") or payload.get("activity") or "unknown").strip().lower()
    if activity not in {
        "coding",
        "reviewing",
        "researching",
        "writing",
        "communication",
        "browsing",
        "planning",
        "system_admin",
        "idle",
        "unknown",
    }:
        activity = "unknown"
    app_guess = payload.get("applications") or payload.get("app_guess")
    if isinstance(app_guess, list):
        applications = [str(item).strip() for item in app_guess if str(item).strip()]
    elif isinstance(app_guess, str) and app_guess.strip():
        applications = [part.strip() for part in re.split(r"\s*(?:,| and )\s*", app_guess) if part.strip()]
    else:
        applications = []
    visible_text = payload.get("key_visible_text") or payload.get("visible_text") or []
    if isinstance(visible_text, str):
        key_visible_text = [visible_text]
    elif isinstance(visible_text, list):
        key_visible_text = [str(item) for item in visible_text]
    else:
        key_visible_text = []
    sensitive = bool(payload.get("sensitive_content_seen") or payload.get("sensitive"))
    return {
        "schema_version": "seraph.screenshot_analysis.v1",
        "prompt_version": "seraph.screenshot_analysis.prompt.v1",
        "summary": str(payload.get("summary") or "Screenshot activity is unclear."),
        "detailed_observations": payload.get("detailed_observations")
        if isinstance(payload.get("detailed_observations"), list)
        else [],
        "activity_type": activity,
        "project": payload.get("project") if isinstance(payload.get("project"), str) else None,
        "applications": applications,
        "visible_artifacts": payload.get("visible_artifacts") if isinstance(payload.get("visible_artifacts"), list) else [],
        "key_visible_text": key_visible_text,
        "user_intent": str(payload.get("user_intent") or "unknown"),
        "goal_alignment": payload.get("goal_alignment")
        if isinstance(payload.get("goal_alignment"), dict)
        else {
            "status": "unknown",
            "goal_refs": [],
            "evidence": [],
            "needle_movement": "unknown",
        },
        "confidence": payload.get("confidence", 0.0),
        "sensitive_content_seen": sensitive,
        "privacy_notes": payload.get("privacy_notes")
        if isinstance(payload.get("privacy_notes"), list)
        else ["VLM wrapper coerced simple screenshot JSON into Seraph's current schema."],
        "report_tags": payload.get("report_tags") if isinstance(payload.get("report_tags"), list) else ["screenshot", activity],
    }


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


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
