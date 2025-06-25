#!/usr/bin/env bash
docker run --rm -it \
  --network=host \
  --ipc=host \
  --gpus=all \
  -v $(pwd):/workspace/cornserve \
  -v $HF_HOME:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e OTEL_SDK_DISABLED=true \
  -e SIDECAR_IS_LOCAL=true \
  -v /dev/shm:/dev/shm \
  -v /sys/class/infiniband:/sys/class/infiniband \
  -v /dev/infiniband:/dev/infiniband \
  --entrypoint=bash \
  cornserve/eric:latest

: '
python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 0 --model.id google/gemma-3-4b-it
python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 0 1 --model.id google/gemma-3-4b-it --model.tp-size 2
python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 0 1 --model.id Qwen/Qwen2.5-VL-7B-Instruct --model.tp-size 2
python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 0 1 --model.id Qwen/Qwen2.5-Omni-7B --model.tp-size 2
python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 0 1 --model.id llava-hf/llava-onevision-qwen2-7b-ov-chat-hf --model.tp-size 2


CUDA_VISIBLE_DEVICES=0 python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 0 --model.id Qwen/Qwen2.5-Omni-7B --model.modality AUDIO --server.port 7999
CUDA_VISIBLE_DEVICES=1 python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 1 --model.id Qwen/Qwen2.5-Omni-7B --model.modality VIDEO --server.port 8000
CUDA_VISIBLE_DEVICES=3 python3 -u -m cornserve.task_executors.eric.entrypoint --sidecar.ranks 3 --model.id Qwen/Qwen2.5-Omni-7B --model.modality IMAGE --server.port 8003

pip install -e .[eric] -e .[audio]
'
