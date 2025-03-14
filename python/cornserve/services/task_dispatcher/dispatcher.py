"""Task Dispatcher."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import httpx

from cornserve.logging import get_logger
from cornserve.services.task_dispatcher.models import TaskInfo
from cornserve.services.pb import task_manager_pb2

logger = get_logger(__name__)


class TaskDispatcher:
    """Task Dispatcher."""

    def __init__(self) -> None:
        """Initialize the Task Dispatcher."""
        self.app_lock = asyncio.Lock()
        self.app_task_info: dict[str, dict[str, TaskInfo]] = {}

        self.ongoing_task_lock = asyncio.Lock()
        self.app_ongoing_invokes: dict[str, list[asyncio.Task]] = defaultdict(list)

    async def nofity_app_registration(self, app_id: str, task_info: list[TaskInfo]) -> None:
        """Register newly spawned task managers to the dispatcher."""
        async with self.app_lock:
            if app_id in self.app_task_info:
                raise ValueError(f"App ID {app_id} already exists in task dispatcher.")
            self.app_task_info[app_id] = {task.id: task for task in task_info}

        logger.info("Registered new app %s with tasks %s", app_id, task_info)

    async def notify_app_unregistration(self, app_id: str) -> None:
        """Remove task managers associated with a task from the dispatcher.

        This is called when an app is unregistered from the dispatcher. Each app keeps
        track of the ongoing invoke tasks. When the app is unregistered, all ongoing invoke
        tasks are cancelled.
        """
        async with self.app_lock:
            if app_id not in self.app_task_info:
                raise ValueError(f"App ID {app_id} not found in task dispatcher.")

            task_infos = self.app_task_info.pop(app_id)

        # Cancel all ongoing invokes for the app.
        async with self.ongoing_task_lock:
            for invoke_task in self.app_ongoing_invokes.pop(app_id, []):
                invoke_task.cancel()

        # Close all channels to the task managers.
        channel_close_coros = []
        for task_info in task_infos.values():
            for task_manager in task_info.task_managers:
                channel_close_coros.append(task_manager.channel.close(grace=1.0))
        await asyncio.gather(*channel_close_coros)

        logger.info("Unregistered app %s", app_id)

    async def shutdown(self) -> None:
        """Shutdown the Task Dispatcher."""

    async def invoke(self, app_id: str, task_id: str, request_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Invoke a task with the given request data."""
        async with self.app_lock:
            try:
                task_infos = self.app_task_info[app_id]
            except KeyError as e:
                raise ValueError(f"App ID {app_id} not found in task dispatcher.") from e

            try:
                task_info = task_infos[task_id]
            except KeyError as e:
                raise ValueError(f"Task ID {task_id} not found for app {app_id}.") from e

        # Spawn invoke task and add it to the list of ongoing invokes for the app.
        invoke_task = asyncio.create_task(self._do_invoke(app_id, task_info, request_id, data))
        async with self.ongoing_task_lock:
            self.app_ongoing_invokes[app_id].append(invoke_task)

        try:
            await invoke_task
        except asyncio.CancelledError as e:
            raise ValueError("Task invoke cancelled. The app is likely shutting down.") from e
        finally:
            async with self.ongoing_task_lock:
                self.app_ongoing_invokes[app_id].remove(invoke_task)

    async def _do_invoke(
        self,
        app_id: str,
        task_info: TaskInfo,
        request_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Do the actual invocation of the task."""
        # Query the task managers for which task executor to route to.
        routing_coros: list[asyncio.Future[task_manager_pb2.GetRouteResponse]] = []
        for task_manager in task_info.task_managers:
            routing_coros.append(
                task_manager.stub.GetRoute(
                    task_manager_pb2.GetRouteRequest(
                        app_id=app_id,
                        request_id=request_id,
                        routing_hint=None,
                    ),
                ),
            )
        task_executors = await asyncio.gather(*routing_coros)

        # Reformat the request data depending on the type of the task.
        task_info.id
