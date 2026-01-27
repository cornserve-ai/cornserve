"""Spins up Geri."""

from __future__ import annotations

import asyncio
import signal

import tyro
import uvicorn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

from cornserve.logging import get_logger
from cornserve.task_executors.geri.config import GeriConfig
from cornserve.task_executors.geri.engine.client import EngineClient
from cornserve.task_executors.geri.router import create_app
from cornserve.tracing import configure_otel

logger = get_logger("cornserve.task_executors.geri.entrypoint")


async def serve(geri_config: GeriConfig) -> None:
    """Serve the Geri model as a FastAPI app."""
    logger.info("Starting Geri with %s", geri_config)

    configure_otel(f"geri{str(geri_config.sidecar.ranks).replace(' ', '')}")

    app = create_app(geri_config)

    FastAPIInstrumentor.instrument_app(app, exclude_spans=["send", "receive"])
    ThreadingInstrumentor().instrument()

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

    config = uvicorn.Config(app, host=geri_config.server.host, port=geri_config.server.port)
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve())

    def shutdown() -> None:
        engine_client: EngineClient = app.state.engine_client
        loop.create_task(engine_client.shutdown())
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("Shutting down FastAPI server.")
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(serve(tyro.cli(GeriConfig)))
