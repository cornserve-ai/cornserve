# Build flash-attn wheel inside the `devel` image which has `nvcc`.
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel AS builder

ARG max_jobs=64
ENV MAX_JOBS=${max_jobs}
ENV NVCC_THREADS=8
RUN pip wheel -w /tmp/wheels --no-build-isolation --no-deps --verbose flash-attn==2.7.4.post1

# Just copy over the flash-attn wheel for eric
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

COPY --from=builder /tmp/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget build-essential librdmacm-dev net-tools && \
    rm -rf /var/lib/apt/lists/*

########### Install UCX 1.18.0 ###########
RUN wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
RUN tar xzf ucx-1.18.0.tar.gz
WORKDIR /workspace/ucx-1.18.0
RUN mkdir build
RUN cd build && \
      ../configure --build=x86_64-unknown-linux-gnu --host=x86_64-unknown-linux-gnu --program-prefix= --disable-dependency-tracking \
      --prefix=/usr --exec-prefix=/usr --bindir=/usr/bin --sbindir=/usr/sbin --sysconfdir=/etc --datadir=/usr/share --includedir=/usr/include \
      --libdir=/usr/lib64 --libexecdir=/usr/libexec --localstatedir=/var --sharedstatedir=/var/lib --mandir=/usr/share/man --infodir=/usr/share/info \
      --disable-logging --disable-debug --disable-assertions --enable-mt --disable-params-check --without-go --without-java --enable-cma \
      --with-verbs --with-mlx5 --with-rdmacm --without-rocm --with-xpmem --without-fuse3 --without-ugni --without-mad --without-ze && \
      make -j$(nproc) && make install

ENV RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# UCX logging
ENV UCX_LOG_LEVEL=trace
# UCX_LOG_LEVEL to be one of: fatal, error, warn, info, debug, trace, req, data, async, func, poll

# UCX transports
ENV UCX_TLS=rc,ib,tcp
########### End Install UCX ###########

ADD ./python /workspace/cornserve/python
ADD ./third_party /workspace/cornserve/third_party
WORKDIR /workspace/cornserve/python

RUN pip install -e '.[dev]'

# UCXX logging
ENV UCXPY_LOG_LEVEL=ERROR
# python log level syntax: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Disable OpenTelemetry
ENV OTEL_SDK_DISABLED=true
# use local sidecar
ENV SIDECAR_IS_LOCAL=true

WORKDIR /workspace/cornserve
CMD ["bash"]
