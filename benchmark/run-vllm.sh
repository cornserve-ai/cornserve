#!/usr/bin/env bash
# Usage: REGISTRY=<registry_url> ./run-vllm.sh <gpu_index>
# Runs Qwen2.5-VL on GPU <gpu_index> and port 8001+<gpu_index>.
# The image will be pulled from $REGISTRY.  Example:
#   export REGISTRY=myregstry:5000
#   ./run-vllm.sh 0

# GPU index (0-indexed) --------------------------------------------------------
IDX=${1:-}
if [[ -z "$IDX" ]]; then
  echo "Usage: $0 <gpu_index>"
  exit 1
fi

# Registry ---------------------------------------------------------------------
if [[ -z "$REGISTRY" ]]; then
  echo "Error: REGISTRY environment variable not set."
  echo "Please set it, e.g.  export REGISTRY=docker.io"
  exit 1
fi

# Derived parameters -----------------------------------------------------------
PORT=$((8001 + IDX))
IMAGE="${REGISTRY}/cornserve/vllm:latest"

# Run container ----------------------------------------------------------------
docker run --rm -it \
  --network host \
  --ipc host \
  --gpus "device=${IDX}" \
  --pull always \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e OTEL_SDK_DISABLED=true \
  -e SIDECAR_IS_LOCAL=true \
  -v /dev/shm:/dev/shm \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v /dev/infiniband:/dev/infiniband \
  "${IMAGE}" \
  Qwen/Qwen2.5-VL-7B-Instruct --no-enable-prefix-caching --port "${PORT}"
