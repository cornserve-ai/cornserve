"""Spins up the Task Dispatcher service with gRPC and FastAPI."""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING

import uvicorn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from cornserve.logging import get_logger
from cornserve.services.cr_manager.manager import CRManager
from cornserve.services.task_dispatcher.grpc import create_server
from cornserve.services.task_dispatcher.router import create_app
from cornserve.tracing import configure_otel

if TYPE_CHECKING:
    from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher

logger = get_logger("cornserve.services.task_dispatcher.entrypoint")


async def serve() -> None:
    """Serve the Task Dispatcher service."""
    logger.info("Starting Gateway service")

    configure_otel("task_dispatcher")

    # Start CR watcher to load tasks/executors from CRs before the server starts
    logger.info("Starting CR watcher for Task Dispatcher service")
    cr_manager = CRManager()
    cr_watcher_task = asyncio.create_task(
        cr_manager.watch_cr_updates(),
        name="task_dispatcher_cr_watcher"
    )

    # FastAPI server
    app = create_app()

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    GrpcInstrumentorClient().instrument()
    GrpcInstrumentorServer().instrument()

    logger.info("Available HTTP routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info(
            "%s %s",
            list(methods)[0] if len(methods) == 1 else "{" + ",".join(methods) + "}",
            path,
        )

    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    uvicorn_server = uvicorn.Server(config)
    dispatcher: TaskDispatcher = app.state.dispatcher

    # gRPC server
    grpc_server = create_server(dispatcher, cr_manager)

    loop = asyncio.get_running_loop()
    uvicorn_server_task = loop.create_task(uvicorn_server.serve())
    await grpc_server.start()

    def shutdown() -> None:
        uvicorn_server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await uvicorn_server_task
    except asyncio.CancelledError:
        logger.info("Shutting down Task Dispatcher service")
        
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
        
        await dispatcher.shutdown()
        await uvicorn_server.shutdown()
        await grpc_server.stop(5)
        logger.info("Task Dispatcher service shutdown complete")


if __name__ == "__main__":
    asyncio.run(serve())
