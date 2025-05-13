"""Entrypoint for the Task Manager service."""

from __future__ import annotations

import asyncio
import signal

from cornserve.logging import get_logger
from cornserve.services.task_manager.grpc import create_server

logger = get_logger("cornserve.services.task_manager.entrypoint")


async def serve() -> None:
    """Serve the Task Manager service."""
    logger.info("Starting Task Manager service")

    server, servicer = create_server()
    await server.start()

    logger.info("gRPC server started")

    loop = asyncio.get_running_loop()
    server_task = asyncio.create_task(server.wait_for_termination())

    def shutdown() -> None:
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Shutting down Task Manager service")
        await server.stop(5)
        if servicer.manager is not None:
            logger.info("Shutting down task manager...")
            await servicer.manager.shutdown()
        logger.info("Task Manager service shutdown complete")


if __name__ == "__main__":
    asyncio.run(serve())
