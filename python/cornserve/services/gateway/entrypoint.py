"""Spins up the Gateway service."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import TYPE_CHECKING

import uvicorn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorClient
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from cornserve.logging import get_logger
from cornserve.services.gateway.router import create_app
from cornserve.tracing import configure_otel

if TYPE_CHECKING:
    from cornserve.services.gateway.app.manager import AppManager

logger = get_logger("cornserve.services.gateway.entrypoint")


async def serve() -> None:
    """Serve the Gateway as a FastAPI app."""
    logger.info("Starting Gateway service")

    configure_otel("gateway")

    # Start CR watcher to load tasks from Custom Resources BEFORE starting FastAPI
    from cornserve.services.cr_manager.manager import CRManager
    logger.info("Starting CR watcher for Gateway service")
    cr_manager = CRManager()
    
    # Deploy built-in task definitions as Custom Resources
    from cornserve.services.gateway.router import deploy_builtin_task_crds
    logger.info("Deploying built-in task definitions")
    await deploy_builtin_task_crds()
    
    cr_watcher_task = asyncio.create_task(
        cr_manager.watch_cr_updates(),
        name="gateway_cr_watcher"
    )

    app = create_app()
    FastAPIInstrumentor.instrument_app(app)
    GrpcAioInstrumentorClient().instrument()
    HTTPXClientInstrumentor().instrument()

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

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Shutting down Gateway service")
        
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
        
        await app_manager.shutdown()
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(serve())
