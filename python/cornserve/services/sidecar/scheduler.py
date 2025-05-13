from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Type

@dataclass
class _Job:
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    fut: asyncio.Future

# ── async no‑op context manager ─────────────────────────────────────
class _AsyncNullCM:
    """`async with _ASYNC_NULL:` does nothing (like contextlib.nullcontext)."""

    async def __aenter__(self) -> None:            # noqa: D401
        return None

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False                                # don't swallow exceptions

_ASYNC_NULL = _AsyncNullCM()       # singleton instance

class Scheduler:
    """
    Central launch‑controller.
    • Only the *launch order* is serialised.
    • Each job runs concurrently in its own task.
    • Optional `max_concurrency` limits jobs in flight.
    """

    def __init__(self, max_concurrency: int | None = None) -> None:
        self._q: asyncio.Queue[_Job] = asyncio.Queue()
        self._runner_task: asyncio.Task | None = None
        self._sema = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    # ── public API ──────────────────────────────────────────────────
    async def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._q.put(_Job(fn, args, kwargs, fut))
        return await fut                            # caller waits here

    async def schedule(self) -> _Job:
        return await self._q.get()                  # FIFO

    # ── internals ───────────────────────────────────────────────────
    async def _runner(self) -> None:
        while True:
            job = await self.schedule()             # decide what to launch

            async def _execute(j: _Job) -> None:
                cm = self._sema or _ASYNC_NULL      # semaphore or no‑op ctx
                async with cm:
                    try:
                        res = j.fn(*j.args, **j.kwargs)
                        if asyncio.iscoroutine(res):
                            res = await res
                        j.fut.set_result(res)
                    except Exception as exc:
                        j.fut.set_exception(exc)

            asyncio.create_task(_execute(job))      # launch & return to queue

    # ── lifecycle helpers ───────────────────────────────────────────
    def start(self) -> None:
        if self._runner_task is None:
            self._runner_task = asyncio.create_task(self._runner())

    async def stop(self) -> None:
        if self._runner_task:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
