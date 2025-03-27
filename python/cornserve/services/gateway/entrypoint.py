"""Spins up the Gateway service."""

import asyncio
import signal
from typing import TYPE_CHECKING

import uvicorn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

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

    app = create_app()
    FastAPIInstrumentor.instrument_app(app)

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
        await app_manager.shutdown()
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(serve())
