FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime AS base

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        infiniband-diags \
        ibverbs-utils \
        rdma-core \
        ibverbs-providers \
        librdmacm-dev \
        build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python 3.11 --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./python /workspace/cornserve/python
COPY ./third_party/vllm /workspace/cornserve/third_party/vllm
WORKDIR /workspace/cornserve/third_party/vllm

RUN cd ../.. && uv pip install './python[sidecar-api]'

# Install vLLM requirements and clean up cache
RUN uv pip install -r requirements/common.txt \
    && uv pip install -r requirements/cuda.txt \
    && uv cache clean

# Set environment variables
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1.dev
ENV VLLM_USE_PRECOMPILED=1
ENV VLLM_COMMIT=c18b3b8e8bdcebaa150311d2c4911a6428480162
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# Intermediate vllm stage without audio
FROM base AS vllm
RUN uv pip install -e . && uv cache clean

ENV VLLM_USE_V1=1
ENV HF_HOME="/root/.cache/huggingface"
ENTRYPOINT ["vllm", "serve"]

ENV UCX_TLS=cuda,rc,ib
ENV UCX_LOG_LEVEL=debug

# Default final stage with audio support
FROM vllm AS vllm-audio
RUN uv pip install -e .[audio] && uv cache clean
