"""Entrypoint for HuggingFace task executor."""

from __future__ import annotations

import asyncio
import signal
import sys

import uvicorn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import TaskType
from cornserve.task_executors.huggingface.config import HuggingFaceConfig, ModelConfig, ServerConfig
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
    service_name = f"huggingface-{config.task_type.value}-{config.model.id.split('/')[-1].lower()}"
    configure_otel(service_name)

    # Create FastAPI app
    app = create_app(config)

    # Instrument with OpenTelemetry
    FastAPIInstrumentor().instrument_app(app)
    ThreadingInstrumentor().instrument()

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


def main() -> None:
    """Main entrypoint function."""
    # Parse command line arguments
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python -m cornserve.task_executors.huggingface.entrypoint "
            "--task-type <type> --model.id <id> [options]"
        )
        sys.exit(1)

    # Parse arguments using tyro
    # We'll build the config from individual args since tyro doesn't handle nested dataclasses well
    args = sys.argv[1:]

    # Parse task type
    task_type_idx = args.index("--task-type") if "--task-type" in args else -1
    if task_type_idx == -1:
        logger.error("--task-type is required")
        sys.exit(1)
    task_type = TaskType(args[task_type_idx + 1])

    # Parse model id
    model_id_idx = args.index("--model.id") if "--model.id" in args else -1
    if model_id_idx == -1:
        logger.error("--model.id is required")
        sys.exit(1)
    model_id = args[model_id_idx + 1]

    # Parse optional arguments
    port = 8000
    if "--server.port" in args:
        port_idx = args.index("--server.port")
        port = int(args[port_idx + 1])

    max_batch_size = 1
    if "--server.max-batch-size" in args:
        batch_size_idx = args.index("--server.max-batch-size")
        max_batch_size = int(args[batch_size_idx + 1])

    # Create config
    config = HuggingFaceConfig(
        task_type=task_type,
        model=ModelConfig(id=model_id, max_batch_size=max_batch_size),
        server=ServerConfig(host="0.0.0.0", port=port),
    )

    # Run the server
    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
