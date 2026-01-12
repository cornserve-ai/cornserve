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
    "CONTAINER_IMAGE_GERI": lambda: _build_image_name("geri"),
    "CONTAINER_IMAGE_VLLM": lambda: _build_image_name("vllm"),
    "CONTAINER_IMAGE_VLLM_OMNI_TALKER": lambda: _build_image_name("vllm"),
    "CONTAINER_IMAGE_HUGGINGFACE": lambda: _build_image_name("huggingface"),
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
K8S_TASK_DISPATCHER_HEADLESS_SERVICE = "task-dispatcher-headless"
K8S_RESOURCE_MANAGER_GRPC_URL = "resource-manager:50051"
K8S_OTEL_GRPC_URL = "http://jaeger-collector.cornserve-system.svc.cluster.local:4317"
K8S_TASK_EXECUTOR_SECRET_NAME = "cornserve-env"
K8S_TASK_EXECUTOR_HF_TOKEN_KEY = "hf-token"
K8S_TASK_EXECUTOR_HEALTHY_TIMEOUT = 20 * 60.0

# Control-plane sync timeout
SYNC_WATCHERS_TIMEOUT = 30.0  # seconds
SYNC_WATCHERS_POLL_INTERVAL = 0.1  # seconds

# Volume host paths.
VOLUME_HF_CACHE = "/data/hfcache"
VOLUME_SHM = "/dev/shm"
VOLUME_VLLM_EXECUTOR_CACHE = "/data/cornserve/cache"

# Container's internal directory where tasklib modules are written for dynamic import.
TASKLIB_DIR = "/tmp/cornserve_tasklib"

# Unit task profiles mounted here with a ConfigMap.
UNIT_TASK_PROFILES_DIR = "/etc/cornserve/profiles"
K8S_UNIT_TASK_PROFILES_CONFIG_MAP_NAME = "cornserve-profiles"

# Container images name construction.
if TYPE_CHECKING:
    CONTAINER_IMAGE_TASK_MANAGER: str
    CONTAINER_IMAGE_SIDECAR: str
    CONTAINER_IMAGE_ERIC: str
    CONTAINER_IMAGE_GERI: str
    CONTAINER_IMAGE_VLLM: str
    CONTAINER_IMAGE_VLLM_OMNI_TALKER: str
    CONTAINER_IMAGE_PULL_POLICY: str


# CRD constants.
CRD_GROUP = "cornserve.ai"
# Note this is the version of CRD definition, where we only define v1
CRD_VERSION = "v1"

# CR plural names (must match the spec.names.plural in CRD files)
CRD_PLURAL_TASK_DEFINITIONS = "taskdefinitions"
CRD_PLURAL_UNIT_TASK_INSTANCES = "unittaskinstances"
CRD_PLURAL_EXECUTION_DESCRIPTORS = "executiondescriptors"
CRD_PLURAL_LATEST_TASKLIB_RVS = "latesttasklibrvs"

# CR kind names (must match spec.names.kind in CRD files)
CRD_KIND_TASK_DEFINITION = "TaskDefinition"
CRD_KIND_UNIT_TASK_INSTANCE = "UnitTaskInstance"
CRD_KIND_EXECUTION_DESCRIPTOR = "ExecutionDescriptor"
CRD_KIND_LATEST_TASKLIB_RV = "LatestTasklibRV"

# Singleton CR storing the resourceVersion associated with the latest tasklib deployment.
CR_NAME_LATEST_TASKLIB_RV = "latest-tasklib-rv"
CR_KEY_MAX_TASK_CLASS_RV = "maxTaskClassRV"
CR_KEY_MAX_DESCRIPTOR_RV = "maxDescriptorRV"
