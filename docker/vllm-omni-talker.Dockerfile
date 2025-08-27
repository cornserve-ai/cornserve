FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
      && apt-get install -y git curl wget ffmpeg libsndfile1 \
      && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python 3.11 --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ADD . /workspace/cornserve
WORKDIR /workspace/cornserve/python
RUN uv pip install '-e .[sidecar-api, eric-no-gpu]'

WORKDIR /workspace
ENV VLLM_USE_PRECOMPILED=1
ENV VLLM_COMMIT=b99733d0929aba5ec4f523885d9b417d50b90fc2
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1.dev

# Toggle between local vllm and remote for development

# COPY ./third_party/vllm /workspace/c-vllm
# RUN cd c-vllm && \
RUN git clone -b jm-omni-talker-vocoder-v0 https://github.com/cornserve-ai/vllm.git c-vllm && \
  cd c-vllm && \
    uv pip install setuptools_scm torchdiffeq resampy x_transformers && \
    uv pip install -r requirements/cuda.txt && \
    uv pip install --upgrade setuptools wheel && \
    uv pip install "git+https://github.com/huggingface/transformers.git@43bb4c0456ebab67ca6b11fa5fa4c099fb2e6a2c" && \
    uv pip install networkx==3.1 \
        xformers==0.0.29.post2 \
        accelerate \
        qwen-omni-utils \
        modelscope_studio \
        gradio==5.23.1 \
        gradio_client==1.8.0 \
        librosa==0.11.0 \
        ffmpeg==1.4 \
        ffmpeg-python==0.2.0 \
        soundfile==0.13.1 \
        av && \
    uv pip install -e .

ENV HF_HOME="/root/.cache/huggingface"
ENV VLLM_USE_V1=0
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.omni_talker_server"]

