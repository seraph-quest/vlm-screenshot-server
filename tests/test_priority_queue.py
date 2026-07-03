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
        self.assertEqual(payload["max_size"], settings.queue_max_size)
        self.assertEqual(payload["workers"], _effective_queue_workers())
        self.assertEqual(payload["configured_workers"], settings.queue_workers)


if __name__ == "__main__":
    unittest.main()
