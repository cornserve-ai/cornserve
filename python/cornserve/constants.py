"""Constants used throughout Cornserve."""

K8S_NAMESPACE = "cornserve"
K8S_SIDECAR_SERVICE_NAME = "sidecar"
K8S_GATEWAY_SERVICE_HTTP_URL = "http://gateway:8000"
K8S_TASK_DISPATCHER_HTTP_URL = "http://task-dispatcher:8000"
K8S_TASK_DISPATCHER_GRPC_URL = "task-dispatcher:50051"
K8S_RESOURCE_MANAGER_GRPC_URL = "resource-manager:50051"
K8S_OTEL_GRPC_URL = "http://jaeger-collector.cornserve-system:4317"

CONTAINER_IMAGE_TASK_MANAGER = "cornserve/task-manager:latest"
CONTAINER_IMAGE_SIDECAR = "cornserve/sidecar:latest"
CONTAINER_IMAGE_ERIC = "cornserve/eric:latest"
CONTAINER_IMAGE_VLLM = "cornserve/vllm:latest"

# Path on host.
VOLUME_HF_CACHE = "/data/hfcache"
VOLUME_SHM = "/dev/shm"
