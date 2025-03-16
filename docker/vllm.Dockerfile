FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update -y \
      && apt-get install -y git curl wget \
      && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python 3.11 --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ADD . /workspace/cornserve
WORKDIR /workspace/cornserve

RUN uv pip install -r third_party/vllm/requirements-common.txt
RUN uv pip install -r third_party/vllm/requirements-cuda.txt
RUN export VLLM_COMMIT=e02883c40086bb7e99903863a98c8786af2db2fd \
      && export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
      && uv pip install --editable third_party/vllm

# Install CORNSERVE sidecars
RUN pip3 install './python[sidecar]'

ENV VLLM_USE_V1=1
ENTRYPOINT ["vllm", "serve"]
# Below entrypoint is not working
# ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

