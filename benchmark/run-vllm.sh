#!/usr/bin/env bash

# Port is first arg, exit if not provided
PORT=${1:-}
if [ -z "$PORT" ]; then
  echo "Usage: $0 <port>"
  exit 1
fi

docker run --rm -it \
  --network=host \
  --ipc=host \
  --gpus=all \
  --pull=always \
  -v $HF_HOME:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e OTEL_SDK_DISABLED=true \
  -e SIDECAR_IS_LOCAL=true \
  -v /dev/shm:/dev/shm \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v /dev/infiniband:/dev/infiniband \
  docker.io/cornserve/vllm:latest \
  Qwen/Qwen2.5-VL-7B-Instruct --no-enable-prefix-caching --port $PORT

