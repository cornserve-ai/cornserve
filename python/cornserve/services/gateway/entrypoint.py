"""Spins up the Gateway service."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import TYPE_CHECKING

import uvicorn
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorClient

from cornserve.logging import get_logger
from cornserve.services.gateway.router import create_app
from cornserve.services.task_registry import TaskRegistry
from cornserve.tracing import configure_otel
from cornserve.utils import set_ulimit

if TYPE_CHECKING:
    from cornserve.services.gateway.app.manager import AppManager

logger = get_logger("cornserve.services.gateway.entrypoint")


async def serve() -> None:
    """Serve the Gateway as a FastAPI app."""
    logger.info("Starting Gateway service")

    set_ulimit()
    configure_otel("gateway")

    GrpcAioInstrumentorClient().instrument()
    AioHttpClientInstrumentor().instrument()

    app = create_app()

    # Start task watcher to load tasks from Custom Resources BEFORE starting FastAPI
    logger.info("Starting task watcher for Gateway service")
    task_registry: TaskRegistry = app.state.task_registry
    cr_watcher_task = asyncio.create_task(task_registry.watch_updates(), name="gateway_cr_watcher")
    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])

    logger.info("Available routes are:")
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
    server = uvicorn.Server(config)
    app_manager: AppManager = app.state.app_manager

    # `TaskContext` reads this environment variable to determine the URL of the Gateway.
    os.environ["CORNSERVE_GATEWAY_URL"] = "http://localhost:8000"

    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def shutdown() -> None:
        server_task.cancel()
        cr_watcher_task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Shutting down Gateway service")

        # Cancel task watcher task
        if not cr_watcher_task.done():
            logger.info("Cancelling task watcher task")
            cr_watcher_task.cancel()
            try:
                await cr_watcher_task
            except asyncio.CancelledError:
                logger.info("task watcher task cancelled successfully")

        # Close CR manager
        await task_registry.shutdown()

        await app_manager.shutdown()
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(serve())
