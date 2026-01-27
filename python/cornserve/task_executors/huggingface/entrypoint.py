"""Entrypoint for HuggingFace task executor."""

from __future__ import annotations

import asyncio
import signal

import tyro
import uvicorn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.config import HuggingFaceConfig
from cornserve.task_executors.huggingface.engine import HuggingFaceEngine
from cornserve.task_executors.huggingface.router import create_app
from cornserve.tracing import configure_otel

logger = get_logger("cornserve.task_executors.huggingface.entrypoint")


async def serve(config: HuggingFaceConfig) -> None:
    """Serve the HuggingFace task executor as a FastAPI app.

    Args:
        config: Configuration for the task executor.
    """
    logger.info("Starting HuggingFace task executor with config: %s", config)

    # Configure OpenTelemetry
    service_name = f"huggingface-{config.model.model_type.value}-{config.model.id.split('/')[-1].lower()}"
    configure_otel(service_name)

    # Create FastAPI app
    app = create_app(config)

    # Instrument with OpenTelemetry
    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])

    # Log available routes
    logger.info("Available routes:")
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

    # Configure uvicorn server
    uvicorn_config = uvicorn.Config(app, host=config.server.host, port=config.server.port, log_level="info")
    server = uvicorn.Server(uvicorn_config)

    # Start server
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def shutdown() -> None:
        """Shutdown handler."""
        logger.info("Received shutdown signal")
        engine: HuggingFaceEngine = app.state.engine
        loop.create_task(engine.shutdown())
        server_task.cancel()

    # Set up signal handlers
    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Server task cancelled, shutting down")
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(serve(tyro.cli(HuggingFaceConfig)))
