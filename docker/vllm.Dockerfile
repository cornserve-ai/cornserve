FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update -y \
      && apt-get install -y git curl wget \
      && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python 3.11 --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ADD . /workspace/cornserve
WORKDIR /workspace/cornserve/third_party/vllm

RUN uv pip install -r requirements-common.txt
RUN uv pip install -r requirements-cuda.txt
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1.dev
RUN export VLLM_COMMIT=e02883c40086bb7e99903863a98c8786af2db2fd \
      && export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
      && uv pip install --editable .

# Install CORNSERVE sidecars
RUN cd ../.. && uv pip install './python[sidecar]'

ENV VLLM_USE_V1=1
ENTRYPOINT ["vllm", "serve"]
