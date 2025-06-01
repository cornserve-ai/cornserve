"""Constants used throughout Cornserve.

Environment variables expected:
- `CORNSERVE_IMAGE_PREFIX`: Docker image prefix (default: "docker.io/cornserve")
- `CORNSERVE_IMAGE_TAG`: Docker image tag (default: "latest")
- `CORNSERVE_IMAGE_PULL_POLICY`: Docker image pull policy (default: "IfNotPresent")

These environment variables are set by different Kustomize overlays depending on
the deployment context (e.g., local, dev, prod).
"""

import os
import warnings
from typing import TYPE_CHECKING, Any


def _get_env_warn_default(var_name: str, default: str) -> str:
    """Get environment variable with a warning if not set, returning a default value."""
    try:
        return os.environ[var_name]
    except KeyError:
        warnings.warn(
            f"Environment variable {var_name} not set, using default '{default}'.",
            stacklevel=2,
        )
        return default


def _build_image_name(name: str) -> str:
    """Builds a full image name with prefix, tag, and pull policy."""
    image_prefix = _get_env_warn_default("CORNSERVE_IMAGE_PREFIX", "docker.io/cornserve").strip("/")
    image_tag = _get_env_warn_default("CORNSERVE_IMAGE_TAG", "latest")
    return f"{image_prefix}/{name}:{image_tag}"


# Cache for lazy-loaded constants
_lazy_cache = {}

# Define which constants should be lazily loaded
_LAZY_CONSTANTS = {
    "CONTAINER_IMAGE_TASK_MANAGER": lambda: _build_image_name("task-manager"),
    "CONTAINER_IMAGE_SIDECAR": lambda: _build_image_name("sidecar"),
    "CONTAINER_IMAGE_ERIC": lambda: _build_image_name("eric"),
    "CONTAINER_IMAGE_VLLM": lambda: _build_image_name("vllm"),
    "CONTAINER_IMAGE_PULL_POLICY": lambda: _get_env_warn_default("CORNSERVE_IMAGE_PULL_POLICY", "IfNotPresent"),
}


def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy loading of image-related constants."""
    if name in _LAZY_CONSTANTS:
        if name not in _lazy_cache:
            _lazy_cache[name] = _LAZY_CONSTANTS[name]()
        return _lazy_cache[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Kubernetes resources.
K8S_NAMESPACE = "cornserve"
K8S_CORNSERVE_CONFIG_MAP_NAME = "cornserve-config"
K8S_SIDECAR_SERVICE_NAME = "sidecar"
K8S_GATEWAY_SERVICE_HTTP_URL = "http://gateway:8000"
K8S_TASK_DISPATCHER_HTTP_URL = "http://task-dispatcher:8000"
K8S_TASK_DISPATCHER_GRPC_URL = "task-dispatcher:50051"
K8S_RESOURCE_MANAGER_GRPC_URL = "resource-manager:50051"
K8S_OTEL_GRPC_URL = "http://jaeger-collector.cornserve-system.svc.cluster.local:4317"
K8S_TASK_EXECUTOR_SECRET_NAME = "cornserve-env"
K8S_TASK_EXECUTOR_HF_TOKEN_KEY = "hf-token"
K8S_TASK_EXECUTOR_HEALTHY_TIMEOUT = 20 * 60.0

# Volume host paths.
VOLUME_HF_CACHE = "/data/hfcache"
VOLUME_SHM = "/dev/shm"

# Container images name construction.
if TYPE_CHECKING:
    CONTAINER_IMAGE_TASK_MANAGER: str
    CONTAINER_IMAGE_SIDECAR: str
    CONTAINER_IMAGE_ERIC: str
    CONTAINER_IMAGE_VLLM: str
    CONTAINER_IMAGE_PULL_POLICY: str
