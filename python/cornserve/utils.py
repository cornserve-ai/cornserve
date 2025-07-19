from cornserve.logging import get_logger

logger = get_logger(__name__)

# Adapted from: https://github.com/vllm-project/vllm/blob/8a8fc946398c34a3b23786c9cb7bf217e223b268/vllm/utils/__init__.py#L2725
def set_ulimit(target_soft_limit=65535):
    import resource
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type,
                               (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n", current_soft, e)
