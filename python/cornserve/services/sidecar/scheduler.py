"""Template Scheduler for Sidecar."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any


@dataclass
class _Job:
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    fut: asyncio.Future


# ── async no‑op context manager ─────────────────────────────────────
class _AsyncNullCM:
    """`async with _ASYNC_NULL:` does nothing (like contextlib.nullcontext)."""

    async def __aenter__(self) -> None:  # noqa: D401
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


_ASYNC_NULL = _AsyncNullCM()


class Scheduler:
    """Central launch‑controller."""

    def __init__(self, max_concurrency: int | None = None) -> None:
        """Initialize the scheduler."""
        self._q: asyncio.Queue[_Job] = asyncio.Queue()
        self._runner_task: asyncio.Task | None = None
        self._sema = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Submit a job to the queue."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._q.put(_Job(fn, args, kwargs, fut))
        return await fut

    async def schedule(self) -> _Job:
        """Schedule the next job in the queue."""
        return await self._q.get()

    async def _runner(self) -> None:
        """Infinite loop to process jobs in the queue."""
        while True:
            job = await self.schedule()

            async def _execute(j: _Job) -> None:
                cm = self._sema or _ASYNC_NULL
                async with cm:
                    try:
                        res = j.fn(*j.args, **j.kwargs)
                        if asyncio.iscoroutine(res):
                            res = await res
                        j.fut.set_result(res)
                    except Exception as exc:
                        j.fut.set_exception(exc)

            asyncio.create_task(_execute(job))

    def start(self) -> None:
        """Start the scheduler and begin processing jobs."""
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._runner())

    async def stop(self) -> None:
        """Stop the scheduler and cancel all jobs in flight."""
        if self._runner_task:
            self._runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
