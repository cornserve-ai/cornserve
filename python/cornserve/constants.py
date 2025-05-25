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
    return f"{_image_prefix}/{name}:{_image_tag}"


_image_prefix = _get_env_warn_default("CORNSERVE_IMAGE_PREFIX", "docker.io/cornserve").strip("/")
_image_pull_policy = _get_env_warn_default("CORNSERVE_IMAGE_PULL_POLICY", "IfNotPresent")
_image_tag = _get_env_warn_default("CORNSERVE_IMAGE_TAG", "latest")


# Kubernetes resources.
K8S_NAMESPACE = "cornserve"
K8S_CORNSERVE_CONFIG_MAP_NAME = "cornserve-config"
K8S_SIDECAR_SERVICE_NAME = "sidecar"
K8S_GATEWAY_SERVICE_HTTP_URL = "http://gateway:8000"
K8S_TASK_DISPATCHER_HTTP_URL = "http://task-dispatcher:8000"
K8S_TASK_DISPATCHER_GRPC_URL = "task-dispatcher:50051"
K8S_RESOURCE_MANAGER_GRPC_URL = "resource-manager:50051"
K8S_OTEL_GRPC_URL = "http://jaeger-collector.cornserve-system.svc.cluster.local:4317"

# Container images.
CONTAINER_IMAGE_TASK_MANAGER = _build_image_name("task-manager")
CONTAINER_IMAGE_SIDECAR = _build_image_name("sidecar")
CONTAINER_IMAGE_ERIC = _build_image_name("eric")
CONTAINER_IMAGE_VLLM = _build_image_name("vllm")
CONTAINER_IMAGE_PULL_POLICY = _image_pull_policy

# Path on host.
VOLUME_HF_CACHE = "/data/hfcache"
VOLUME_SHM = "/dev/shm"
