FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get upgrade -y
RUN apt-get install pciutils ibverbs-utils infiniband-diags libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev wget build-essential -y

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install UCX
RUN wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
RUN tar xzf ucx-1.18.0.tar.gz
WORKDIR /workspace/ucx-1.18.0
RUN mkdir build
RUN cd build && \
      ../configure --build=x86_64-redhat-linux-gnu --host=x86_64-redhat-linux-gnu --program-prefix= --disable-dependency-tracking \
      --prefix=/usr --exec-prefix=/usr --bindir=/usr/bin --sbindir=/usr/sbin --sysconfdir=/etc --datadir=/usr/share --includedir=/usr/include \
      --libdir=/usr/lib64 --libexecdir=/usr/libexec --localstatedir=/var --sharedstatedir=/var/lib --mandir=/usr/share/man --infodir=/usr/share/info \
      --disable-logging --disable-debug --disable-assertions --enable-mt --disable-params-check --without-go --without-java --enable-cma \
      --with-verbs --with-mlx5 --with-rdmacm --without-rocm --with-xpmem --without-fuse3 --without-ugni --without-mad --without-ze && \
      make -j$(nproc) && make install

# ENV UCX_LOG_LEVEL=trace
ENV RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true
ENV UCX_TLS=rc,ib,sm,shm,mm

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e '.[sidecar]'

ENTRYPOINT ["python", "-u", "-m", "cornserve.services.sidecar.server"]
