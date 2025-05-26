FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

RUN apt-get update -y \
      && apt-get install -y git curl wget \
      && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python 3.11 --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ADD . /workspace/cornserve
WORKDIR /workspace/cornserve/third_party/vllm

# Install CORNSERVE sidecars
RUN cd ../.. && uv pip install './python[sidecar-api]'

RUN uv pip install -r requirements/common.txt
RUN uv pip install -r requirements/cuda.txt
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1.dev
ENV VLLM_USE_PRECOMPILED=1
RUN export VLLM_COMMIT=81712218341ce09d555579829e8903e7a9aa4880 \
      && export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
      && uv pip install -e . -e .[audio]


ENV VLLM_USE_V1=1
ENV HF_HOME="/root/.cache/huggingface"
ENTRYPOINT ["vllm", "serve"]
