"""Utility functions."""

from __future__ import annotations

import resource

from grpc.aio import AioRpcError

from cornserve.logging import get_logger

logger = get_logger(__name__)


# Adapted from: https://github.com/vllm-project/vllm/blob/8a8fc946398c34a3b23786c9cb7bf217e223b268/vllm/utils/__init__.py#L2725
def set_ulimit(target_soft_limit=65535):
    """Set the soft limit for the number of open file descriptors."""
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


def format_grpc_error(error: AioRpcError) -> str:
    """Format a gRPC error for better readability."""
    status_code = error.code()
    details = error.details()
    # debug_error_string = error.debug_error_string()
    # encoded_error_string = debug_error_string.encode("utf-8", "ignore")
    # formatted = encoded_error_string.decode("unicode_escape", "ignore")
    # return f"Status Code: {status_code}\n  Details: {details}\n  Debug Error String: {formatted}"
    formatted = f"\nStatus Code: {status_code}\n  Details: {details}"
    formatted = formatted.replace("\n", "\t\n")
    return formatted
