"""Entrypoint for the Task Manager service."""

from __future__ import annotations

import asyncio
import signal

from cornserve.logging import get_logger
from cornserve.services.task_registry import TaskRegistry
from cornserve.services.task_manager.grpc import create_server

logger = get_logger("cornserve.services.task_manager.entrypoint")


async def serve() -> None:
    """Serve the Task Manager service."""
    logger.info("Starting Task Manager service")

    # Start registry watcher to load tasks/descriptors before gRPC server starts
    logger.info("Starting registry watcher for Task Manager service")
    task_registry = TaskRegistry()
    cr_watcher_task = asyncio.create_task(
        task_registry.watch_updates(),
        name="task_manager_entrypoint_cr_watcher"
    )

    server, servicer = create_server(task_registry)
    await server.start()

    logger.info("gRPC server started")

    loop = asyncio.get_running_loop()
    server_task = asyncio.create_task(server.wait_for_termination())

    def shutdown() -> None:
        server_task.cancel()
        cr_watcher_task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Shutting down Task Manager service")
        
        # Cancel CR watcher
        if not cr_watcher_task.done():
            logger.info("Cancelling CR watcher task")
            cr_watcher_task.cancel()
            try:
                await cr_watcher_task
            except asyncio.CancelledError:
                logger.info("CR watcher task cancelled successfully")
        
        # Close registry
        await task_registry.shutdown()
        
        await server.stop(5)
        if servicer.manager is not None:
            logger.info("Shutting down task manager...")
            await servicer.manager.shutdown()
        logger.info("Task Manager service shutdown complete")


if __name__ == "__main__":
    asyncio.run(serve())
