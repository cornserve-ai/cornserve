import os
import sys
import logging

from typing import Any, MutableMapping

def get_logger(name: str, adapters: list[type[logging.LoggerAdapter]] = []) -> logging.Logger | logging.LoggerAdapter:
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
    for adapter in adapters:
        logger = adapter(logger)
    return logger


class SidcarAdapter(logging.LoggerAdapter):
    """Adapter that prepends 'Sidecar {rank}' to all messages."""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger, {})
        if pod_name := os.environ.get("SIDECAR_POD_NAME"):
            self.sidecar_rank = int(pod_name.split("-")[-1])
        else:
            self.sidecar_rank = int(os.environ.get("SIDECAR_RANK", -1))
        assert self.sidecar_rank >= 0, "SIDECAR_RANK or SIDECAR_POD_NAME must be set."

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple:
        return f"Sidecar {self.sidecar_rank}: {msg}", kwargs
