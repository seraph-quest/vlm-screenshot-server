import asyncio
import unittest
from unittest.mock import patch

from app import main
from app.main import PriorityWorkQueue, QueuedWork, _priority_rank, _queue_worker, _stream_queued_chat_completion


class _FakeStreamResponse:
    status_code = 200
    text = ""

    def __init__(self, chunks: list[bytes], release_second_chunk: asyncio.Event) -> None:
        self._chunks = chunks
        self._release_second_chunk = release_second_chunk

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        yield self._chunks[0]
        await self._release_second_chunk.wait()
        for chunk in self._chunks[1:]:
            yield chunk


class _FakeAsyncClient:
    def __init__(self, *args, chunks: list[bytes], release_second_chunk: asyncio.Event, **kwargs) -> None:
        self._chunks = chunks
        self._release_second_chunk = release_second_chunk

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def stream(self, *args, **kwargs) -> _FakeStreamResponse:
        return _FakeStreamResponse(self._chunks, self._release_second_chunk)


class ChatStreamingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._original_queue = main._work_queue
        main._work_queue = PriorityWorkQueue(max_size=4, max_background_active=1)
        self._worker = asyncio.create_task(_queue_worker(0))

    async def asyncTearDown(self) -> None:
        self._worker.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self._worker
        main._work_queue = self._original_queue

    async def test_streaming_chat_forwards_chunks_while_worker_remains_active(self) -> None:
        release_second_chunk = asyncio.Event()
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]

        def _fake_client(*args, **kwargs):
            return _FakeAsyncClient(
                *args,
                chunks=chunks,
                release_second_chunk=release_second_chunk,
                **kwargs,
            )

        with patch("app.main.httpx.AsyncClient", _fake_client):
            stream = _stream_queued_chat_completion(
                payload={"model": "local", "stream": True, "messages": []},
                headers={"Content-Type": "application/json"},
                priority="interactive",
            )

            first_chunk = await stream.__anext__()
            self.assertEqual(first_chunk, chunks[0])
            self.assertEqual(main._ensure_queue().active(), 1)

            release_second_chunk.set()
            remaining = [chunk async for chunk in stream]

        self.assertEqual(remaining, chunks[1:])
        self.assertEqual(main._ensure_queue().active(), 0)


class ChatStreamingQueueRejectionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._original_queue = main._work_queue
        main._work_queue = PriorityWorkQueue(max_size=1, max_background_active=1)

    async def asyncTearDown(self) -> None:
        main._work_queue = self._original_queue

    async def test_streaming_chat_reports_queue_clear_without_hanging(self) -> None:
        stream = _stream_queued_chat_completion(
            payload={"model": "local", "stream": True, "messages": []},
            headers={"Content-Type": "application/json"},
            priority="normal",
        )
        first_chunk = asyncio.create_task(stream.__anext__())
        await self._wait_for_queue_size(1)

        cleared = await main._ensure_queue().clear_queued()

        self.assertEqual(cleared, 1)
        chunk = await asyncio.wait_for(first_chunk, timeout=0.1)
        self.assertIn(b'"error":"vlm_queue_cleared"', chunk)
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

    async def test_streaming_chat_reports_priority_preemption_without_hanging(self) -> None:
        stream = _stream_queued_chat_completion(
            payload={"model": "local", "stream": True, "messages": []},
            headers={"Content-Type": "application/json"},
            priority="background",
        )
        first_chunk = asyncio.create_task(stream.__anext__())
        await self._wait_for_queue_size(1)

        await main._ensure_queue().put(
            (
                _priority_rank("interactive"),
                10,
                QueuedWork(
                    label="chat",
                    priority="interactive",
                    submitted_at=0.0,
                    run=lambda: asyncio.sleep(0),
                    future=asyncio.get_running_loop().create_future(),
                ),
            )
        )

        chunk = await asyncio.wait_for(first_chunk, timeout=0.1)
        self.assertIn(b'"error":"vlm_queue_preempted"', chunk)
        with self.assertRaises(StopAsyncIteration):
            await stream.__anext__()

    async def _wait_for_queue_size(self, expected: int) -> None:
        for _ in range(20):
            if main._ensure_queue().qsize() == expected:
                return
            await asyncio.sleep(0)
        self.fail(f"queue size did not reach {expected}")


if __name__ == "__main__":
    unittest.main()
