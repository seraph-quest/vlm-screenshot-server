import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app import main


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class AnalyzeRetryTest(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_image_retries_transient_backend_502(self) -> None:
        original_attempts = main.settings.vlm_analyze_attempts
        main.settings.vlm_analyze_attempts = 2
        valid_payload = (
            '{"summary":"ok","visible_text":[],"applications":[],"activity_type":"unknown",'
            '"confidence":0.5}'
        )
        submit = AsyncMock(
            side_effect=[
                HTTPException(status_code=502, detail="VLM backend HTTP 502"),
                _FakeResponse(valid_payload),
            ]
        )
        try:
            with patch("app.main._submit_backend_work", submit), patch("app.main.asyncio.sleep", AsyncMock()):
                result = await main._analyze_image(
                    b"image-bytes",
                    media_type="image/png",
                    app_hint=None,
                    prompt="return json",
                    runtime_profile="screenshot_fast",
                    runtime_path="screenshot_image_analysis",
                    priority="normal",
                    reasoning="off",
                    profile_options=None,
                )
        finally:
            main.settings.vlm_analyze_attempts = original_attempts

        self.assertEqual(submit.await_count, 2)
        self.assertEqual(result.analysis["schema_version"], "seraph.screenshot_analysis.v1")
        self.assertEqual(result.analysis["activity_type"], "unknown")
        self.assertEqual(result.analysis["summary"], "ok")


if __name__ == "__main__":
    unittest.main()
