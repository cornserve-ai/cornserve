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
from cornserve.tracing import ResetOTelContextMiddleware, configure_otel
from cornserve.utils import set_ulimit

if TYPE_CHECKING:
    from cornserve.services.gateway.app.manager import AppManager
    from cornserve.services.gateway.app_registry import AppRegistry
    from cornserve.services.gateway.task_manager import TaskManager

logger = get_logger("cornserve.services.gateway.entrypoint")


async def serve() -> None:
    """Serve the Gateway as a FastAPI app."""
    logger.info("Starting Gateway service")

    set_ulimit()
    configure_otel("gateway")

    GrpcAioInstrumentorClient().instrument()
    AioHttpClientInstrumentor().instrument()

    role = os.environ.get("GATEWAY_ROLE", "master")
    logger.info("Gateway role: %s", role)
    app = create_app(role=role)

    logger.info("Starting task and app watchers for Gateway service")
    task_registry: TaskRegistry = app.state.task_registry
    await task_registry.ensure_latest_tasklib_rv_cr_exists()
    # Start task watcher to load tasks from Custom Resources BEFORE starting FastAPI
    cr_watcher_task = asyncio.create_task(task_registry.watch_updates(), name="gateway_cr_watcher")

    # Start app registry watcher for distributed app state
    app_registry: AppRegistry = app.state.app_registry
    await app_registry.ensure_latest_app_rv_cr_exists()
    app_cr_watcher_task = asyncio.create_task(app_registry.watch_updates(), name="gateway_app_cr_watcher")

    # Start task instance watcher for multi-replica cache coherence
    task_manager: TaskManager = app.state.task_manager
    await task_manager.ensure_latest_unit_task_instance_rv_cr_exists()
    task_instance_watcher_task = asyncio.create_task(
        task_manager.watch_task_instances(), name="gateway_task_instance_watcher"
    )

    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
    asgi_app = ResetOTelContextMiddleware(app)

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

    config = uvicorn.Config(asgi_app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    app_manager: AppManager = app.state.app_manager

    # `TaskContext` reads this environment variable to determine the URL of the Gateway.
    os.environ["CORNSERVE_GATEWAY_URL"] = "http://localhost:8000"

    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def shutdown() -> None:
        server_task.cancel()
        cr_watcher_task.cancel()
        app_cr_watcher_task.cancel()
        task_instance_watcher_task.cancel()

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

        # Cancel app registry watcher task
        if not app_cr_watcher_task.done():
            logger.info("Cancelling app registry watcher task")
            app_cr_watcher_task.cancel()
            try:
                await app_cr_watcher_task
            except asyncio.CancelledError:
                logger.info("app registry watcher task cancelled successfully")

        # Cancel task instance watcher task
        if not task_instance_watcher_task.done():
            logger.info("Cancelling task instance watcher task")
            task_instance_watcher_task.cancel()
            try:
                await task_instance_watcher_task
            except asyncio.CancelledError:
                logger.info("task instance watcher task cancelled successfully")

        # Close task CR manager
        await task_registry.shutdown()

        # Close app registry
        await app_registry.shutdown()

        # Close profile manager
        await app.state.profile_manager.shutdown()

        await app_manager.shutdown()
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(serve())
