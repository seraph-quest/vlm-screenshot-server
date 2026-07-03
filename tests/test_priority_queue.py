import asyncio
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import (
    PriorityWorkQueue,
    QueuedWork,
    app,
    _effective_background_workers,
    _effective_queue_workers,
    _priority_rank,
    settings,
)


async def _noop() -> None:
    return None


def _work(label: str, priority: str) -> QueuedWork:
    return QueuedWork(
        label=label,
        priority=priority,
        submitted_at=0.0,
        run=_noop,
        future=asyncio.get_running_loop().create_future(),
    )


class PriorityWorkQueueTest(unittest.IsolatedAsyncioTestCase):
    async def test_higher_priority_work_is_returned_first(self) -> None:
        queue = PriorityWorkQueue(max_size=4, max_background_active=1)
        await queue.put((_priority_rank("background"), 1, _work("screen", "background")))
        await queue.put((_priority_rank("interactive"), 2, _work("chat", "interactive")))

        _, _, first = await queue.get()
        _, _, second = await queue.get()

        self.assertEqual(first.label, "chat")
        self.assertEqual(second.label, "screen")

    async def test_same_priority_work_is_fifo(self) -> None:
        queue = PriorityWorkQueue(max_size=4, max_background_active=1)
        await queue.put((_priority_rank("normal"), 1, _work("first", "normal")))
        await queue.put((_priority_rank("normal"), 2, _work("second", "normal")))

        _, _, first = await queue.get()
        await queue.task_done(first)
        _, _, second = await queue.get()

        self.assertEqual(first.label, "first")
        self.assertEqual(second.label, "second")

    async def test_higher_priority_work_preempts_lower_priority_when_full(self) -> None:
        queue = PriorityWorkQueue(max_size=1, max_background_active=1)
        background = _work("screen", "background")
        await queue.put((_priority_rank("background"), 1, background))

        chat = _work("chat", "interactive")
        await queue.put((_priority_rank("interactive"), 2, chat))

        _, _, admitted = await queue.get()
        self.assertEqual(admitted.label, "chat")
        self.assertTrue(background.future.done())
        with self.assertRaises(HTTPException) as raised:
            background.future.result()
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.detail["error"], "vlm_queue_preempted")

    async def test_same_priority_work_waits_when_queue_is_full(self) -> None:
        queue = PriorityWorkQueue(max_size=1, max_background_active=1)
        await queue.put((_priority_rank("normal"), 1, _work("first", "normal")))

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(
                queue.put((_priority_rank("normal"), 2, _work("second", "normal"))),
                timeout=0.01,
            )

        self.assertEqual(queue.qsize(), 1)
        _, _, remaining = await queue.get()
        self.assertEqual(remaining.label, "first")
        await queue.task_done(remaining)

    async def test_interactive_work_can_run_while_background_slot_is_busy(self) -> None:
        queue = PriorityWorkQueue(max_size=4, max_background_active=1)
        await queue.put((_priority_rank("background"), 1, _work("screen-1", "background")))
        await queue.put((_priority_rank("background"), 2, _work("screen-2", "background")))

        _, _, first = await queue.get()
        self.assertEqual(first.label, "screen-1")
        self.assertEqual(queue.active(), 1)
        self.assertEqual(queue.active_background(), 1)

        await queue.put((_priority_rank("interactive"), 3, _work("chat", "interactive")))
        _, _, second = await queue.get()

        self.assertEqual(second.label, "chat")
        self.assertEqual(queue.active(), 2)
        self.assertEqual(queue.active_background(), 1)
        await queue.task_done(first)
        self.assertEqual(queue.active(), 1)
        self.assertEqual(queue.active_background(), 0)
        await queue.task_done(second)
        self.assertEqual(queue.active(), 0)

    async def test_single_gpu_next_dispatch_uses_highest_priority_after_active_finishes(self) -> None:
        queue = PriorityWorkQueue(max_size=4, max_background_active=1)
        await queue.put((_priority_rank("background"), 1, _work("screen-active", "background")))
        _, _, active = await queue.get()
        self.assertEqual(active.label, "screen-active")
        self.assertEqual(queue.active(), 1)
        self.assertEqual(queue.active_background(), 1)

        await queue.put((_priority_rank("background"), 2, _work("screen-next", "background")))
        await queue.put((_priority_rank("interactive"), 3, _work("chat-next", "interactive")))
        self.assertEqual(queue.qsize(), 2)

        await queue.task_done(active)
        self.assertEqual(queue.active(), 0)
        self.assertEqual(queue.active_background(), 0)

        _, _, next_work = await queue.get()
        self.assertEqual(next_work.label, "chat-next")
        await queue.task_done(next_work)

        _, _, remaining = await queue.get()
        self.assertEqual(remaining.label, "screen-next")
        await queue.task_done(remaining)

    async def test_clear_queued_excludes_active_work(self) -> None:
        queue = PriorityWorkQueue(max_size=4, max_background_active=1)
        await queue.put((_priority_rank("background"), 1, _work("active", "background")))
        _, _, active = await queue.get()
        queued_one = _work("queued-one", "normal")
        queued_two = _work("queued-two", "normal")
        await queue.put((_priority_rank("normal"), 2, queued_one))
        await queue.put((_priority_rank("normal"), 3, queued_two))

        cleared = await queue.clear_queued()

        self.assertEqual(cleared, 2)
        self.assertEqual(queue.qsize(), 0)
        self.assertEqual(queue.active(), 1)
        self.assertFalse(active.future.done())
        for queued in (queued_one, queued_two):
            self.assertTrue(queued.future.done())
            with self.assertRaises(HTTPException) as raised:
                queued.future.result()
            self.assertEqual(raised.exception.status_code, 409)
            self.assertEqual(raised.exception.detail["error"], "vlm_queue_cleared")

        await queue.task_done(active)
        self.assertEqual(queue.active(), 0)

    async def test_clear_queued_rejects_admission_waiters(self) -> None:
        queue = PriorityWorkQueue(max_size=1, max_background_active=1)
        queued = _work("queued", "normal")
        waiter = _work("waiter", "normal")
        await queue.put((_priority_rank("normal"), 1, queued))
        waiting_put = asyncio.create_task(queue.put((_priority_rank("normal"), 2, waiter)))
        await asyncio.sleep(0)

        cleared = await queue.clear_queued()

        self.assertEqual(cleared, 1)
        with self.assertRaises(HTTPException) as raised:
            await waiting_put
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.detail["error"], "vlm_queue_cleared")
        self.assertTrue(queued.future.done())
        with self.assertRaises(HTTPException) as queued_raised:
            queued.future.result()
        self.assertEqual(queued_raised.exception.detail["error"], "vlm_queue_cleared")
        self.assertFalse(waiter.future.done())
        self.assertEqual(queue.qsize(), 0)

    async def test_clear_queued_notifies_streaming_waiters(self) -> None:
        queue = PriorityWorkQueue(max_size=2, max_background_active=1)
        cleared_labels: list[str] = []
        stream = _work("stream", "interactive")
        stream.on_clear = lambda: cleared_labels.append(stream.label)
        await queue.put((_priority_rank("interactive"), 1, stream))

        cleared = await queue.clear_queued()

        self.assertEqual(cleared, 1)
        self.assertEqual(cleared_labels, ["stream"])
        with self.assertRaises(HTTPException) as raised:
            stream.future.result()
        self.assertEqual(raised.exception.status_code, 409)

    async def test_effective_background_workers_reserve_capacity(self) -> None:
        original_workers = settings.queue_workers
        original_background_workers = settings.queue_background_workers
        try:
            settings.queue_workers = 2
            settings.queue_background_workers = 2
            self.assertEqual(_effective_background_workers(), 1)

            settings.queue_workers = 1
            settings.queue_background_workers = 1
            self.assertEqual(_effective_background_workers(), 1)
        finally:
            settings.queue_workers = original_workers
            settings.queue_background_workers = original_background_workers

    async def test_default_topology_is_single_gpu_serial(self) -> None:
        self.assertEqual(settings.queue_workers, 1)
        self.assertEqual(_effective_queue_workers(), 1)


class QueueStatusEndpointTest(unittest.TestCase):
    def test_queue_status_endpoint_exposes_operator_queue_telemetry(self) -> None:
        with TestClient(app) as client:
            response = client.get("/queue/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["queued"], 0)
        self.assertEqual(payload["active"], 0)
        self.assertEqual(payload["pending_labels"], [])
        self.assertEqual(payload["max_size"], settings.queue_max_size)
        self.assertEqual(payload["workers"], _effective_queue_workers())
        self.assertEqual(payload["configured_workers"], settings.queue_workers)

    def test_clear_queue_endpoint_reports_cleared_count(self) -> None:
        with TestClient(app) as client:
            response = client.post("/queue/clear")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["cleared"], 0)
        self.assertEqual(payload["queued"], 0)
        self.assertEqual(payload["active"], 0)


if __name__ == "__main__":
    unittest.main()
