import asyncio
import unittest

from fastapi import HTTPException

from app.main import PriorityWorkQueue, QueuedWork, _effective_background_workers, _priority_rank, settings


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


if __name__ == "__main__":
    unittest.main()
