#!/usr/bin/env bash
# Usage: ./run-vllm.sh <gpu_index>
# Runs Qwen2.5-VL on GPU i and port 8001+i (i is 0-indexed).

IDX=${1:-}
if [[ -z "$IDX" ]]; then
  echo "Usage: $0 <gpu_index>"
  exit 1
fi

PORT=$((8001 + IDX))

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
  docker.io/cornserve/vllm:latest \
  Qwen/Qwen2.5-VL-7B-Instruct --no-enable-prefix-caching --port "${PORT}"

