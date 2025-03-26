"""Constants used throughout CornServe."""

K8S_NAMESPACE = "cornserve"
K8S_SIDECAR_STATEFULSET_NAME = "sidecar"
K8S_SIDECAR_SERVICE_NAME = "sidecar"
K8S_TASK_DISPATCHER_HTTP_URL = "task-dispatcher:8000"
K8S_TASK_DISPATCHER_GRPC_URL = "task-dispatcher:50051"
K8S_RESOURCE_MANAGER_GRPC_URL = "resource-manager:50051"
K8S_OTEL_GRPC_URL = "http://jaeger-collector.cornserve:4317"

CONTAINER_IMAGE_TASK_MANAGER = "cornserve/task-manager:latest"
CONTAINER_IMAGE_ERIC = "cornserve/eric:latest"
CONTAINER_IMAGE_VLLM = "cornserve/vllm:latest"

VOLUME_HF_CACHE = "/data/hfcache"
