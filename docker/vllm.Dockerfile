# UCX-cuda build requires the devel image
# TODO: Use a multi-stage build to reduce image size
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel AS base

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

########### Install UCX 1.19.0 ###########
RUN wget https://github.com/openucx/ucx/releases/download/v1.19.0/ucx-1.19.0.tar.gz
RUN tar xzf ucx-1.19.0.tar.gz
WORKDIR /workspace/ucx-1.19.0
RUN mkdir build
RUN cd build && \
      ../configure --build=x86_64-unknown-linux-gnu --host=x86_64-unknown-linux-gnu --program-prefix= --disable-dependency-tracking \
      --prefix=/usr --exec-prefix=/usr --bindir=/usr/bin --sbindir=/usr/sbin --sysconfdir=/etc --datadir=/usr/share --includedir=/usr/include \
      --libdir=/usr/lib64 --libexecdir=/usr/libexec --localstatedir=/var --sharedstatedir=/var/lib --mandir=/usr/share/man --infodir=/usr/share/info \
      --disable-logging --disable-debug --disable-assertions --enable-mt --disable-params-check --without-go --without-java --enable-cma \
      --with-cuda=/usr/local/cuda --with-dm --enable-shared \
      --with-verbs --with-mlx5 --with-rdmacm --without-rocm --with-xpmem --without-fuse3 --without-ugni --without-mad --without-ze && \
      make -j$(nproc) && make install

ENV RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true
ENV LD_LIBRARY_PATH=/usr/lib64:/usr/lib64/ucx:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
########### End Install UCX ###########

############ Install uv ############
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
RUN uv venv --python 3.11 --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

############# Install nixl #############
RUN uv pip install --upgrade pip meson-python ninja pybind11
RUN mkdir -p /opt/ucx-shim/{lib,include}
RUN ln -s /usr/lib64        /opt/ucx-shim/lib
RUN ln -s /usr/include      /opt/ucx-shim/include
ENV UCX_PREFIX=/opt/ucx-shim
RUN uv pip install "git+https://github.com/ai-dynamo/nixl@0.6.1" \
  --config-settings=setup-args="-Ducx_path=$UCX_PREFIX" \
  --config-settings=setup-args="-Dinstall_headers=false" \
  --config-settings=setup-args="-Dbuild_docs=false"
########### End Install nixl ###########

COPY ./python /workspace/cornserve/python
# Toggle between local vllm and remote for development
COPY ./third_party/vllm /workspace/cornserve/third_party/c-vllm
# RUN git clone -b jm-pd https://github.com/cornserve-ai/vllm.git /workspace/cornserve/third_party/c-vllm
WORKDIR /workspace/cornserve/third_party/c-vllm

RUN cd ../.. && uv pip install './python[sidecar-api]'

# Install vLLM requirements and clean up cache
RUN uv pip install --prerelease=allow -r requirements/common.txt \
    && uv pip install --prerelease=allow -r requirements/cuda.txt \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --index-strategy unsafe-best-match \
    && uv cache clean

# Set environment variables
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1.dev
ENV VLLM_USE_PRECOMPILED=1
ENV VLLM_COMMIT=00b31a36a2d0de6d197a473280b2304d482714af
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# Intermediate vllm stage without audio
FROM base AS vllm
RUN uv pip install -e . && uv cache clean

ENV VLLM_USE_V1=1
ENV HF_HOME="/root/.cache/huggingface"
ENTRYPOINT ["vllm", "serve"]

# Default final stage with audio support
FROM vllm AS vllm-audio
RUN uv pip install -e .[audio] && uv cache clean
