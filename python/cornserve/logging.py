import os
import sys
import logging

from typing import Any, MutableMapping

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name with some formatting configs."""
    # No need to reconfigure the logger if it was already created
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("CORNSERVE_LOG_LEVEL", logging.INFO))
    formatter = logging.Formatter("%(asctime)s [%(name)s](%(filename)s:%(lineno)d) %(message)s")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class SidcarAdapter(logging.LoggerAdapter):
    """Adapter that prepends 'Sidecar {rank}' to all messages."""

    def __init__(self, logger: logging.Logger, sidecar_rank: int):
        super().__init__(logger, {})
        self.sidecar_rank = sidecar_rank

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple:
        return f"Sidecar {self.sidecar_rank}: {msg}", kwargs
