"""Kubernetes utilities for the Cornserve CLI."""

from __future__ import annotations

from pathlib import Path

from kubernetes import config
from kubernetes.config.config_exception import ConfigException


def load_k8s_config(kube_config_path: Path | None = None) -> None:
    """Load Kubernetes config with fallback chain.

    Args:
        kube_config_path: Optional path to the Kubernetes config file

    Raises:
        RuntimeError: If unable to load any Kubernetes configuration
    """
    if kube_config_path:
        config.load_kube_config(config_file=str(kube_config_path))
        return

    try:
        config.load_incluster_config()
        return
    except ConfigException:
        pass

    try:
        config.load_kube_config()
    except Exception as e:
        raise RuntimeError(f"Failed to load Kubernetes config: {e}") from e
