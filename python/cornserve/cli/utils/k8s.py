"""Kubernetes utilities for the Cornserve CLI."""

from __future__ import annotations

from kubernetes import config
from kubernetes.config.config_exception import ConfigException


def load_k8s_config(
    kube_config_path: str | None = None,
    fallback_config_paths: list[str] | None = None,
) -> None:
    """Load Kubernetes config with fallback chain.

    Args:
        kube_config_path: Optional path to the Kubernetes config file
        fallback_config_paths: Optional list of fallback config paths to try
            if `kube_config_path` is not provided or fails

    Raises:
        RuntimeError: If unable to load any Kubernetes configuration
    """
    if kube_config_path:
        config.load_kube_config(config_file=kube_config_path)
        return

    try:
        config.load_incluster_config()
        return
    except ConfigException:
        pass

    for path in fallback_config_paths or [None]:
        try:
            config.load_kube_config(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Kubernetes config: {e}") from e
