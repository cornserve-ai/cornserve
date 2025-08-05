"""Entrypoint for the Resource Manager service."""

from __future__ import annotations

import asyncio
import signal

from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorClient, GrpcAioInstrumentorServer

from cornserve.logging import get_logger
from cornserve.services.cr_manager.manager import CRManager
from cornserve.services.resource_manager.grpc import create_server
from cornserve.services.resource_manager.manager import ResourceManager
from cornserve.tracing import configure_otel

logger = get_logger("cornserve.services.resource_manager.entrypoint")


async def serve() -> None:
    """Start the gRPC server."""
    configure_otel("resource_manager")

    GrpcAioInstrumentorServer().instrument()
    GrpcAioInstrumentorClient().instrument()

    # Start CR watcher to load tasks/executors from CRs before gRPC server starts
    logger.info("Starting CR watcher for Resource Manager service")
    cr_manager = CRManager()
    cr_watcher_task = asyncio.create_task(
        cr_manager.watch_cr_updates(),
        name="resource_manager_cr_watcher"
    )

    resource_manager = await ResourceManager.init()

    server = create_server(resource_manager, cr_manager)
    await server.start()

    logger.info("gRPC server started")

    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.wait_for_termination())

    def shutdown() -> None:
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Shutting down Resource Manager service")
        
        # Cancel CR watcher task
        if not cr_watcher_task.done():
            logger.info("Cancelling CR watcher task")
            cr_watcher_task.cancel()
            try:
                await cr_watcher_task
            except asyncio.CancelledError:
                logger.info("CR watcher task cancelled successfully")
        
        # Close CR manager
        await cr_manager.close()
        
        await server.stop(5)
        logger.info("Shutting down resource manager...")
        await resource_manager.shutdown()
        logger.info("Resource Manager service shutdown complete")


if __name__ == "__main__":
    asyncio.run(serve())
