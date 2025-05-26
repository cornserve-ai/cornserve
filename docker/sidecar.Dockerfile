FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get upgrade -y
RUN apt-get install wget build-essential librdmacm-dev net-tools -y

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
########### End Install UCX ###########

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e '.[sidecar]'

ENTRYPOINT ["python", "-u", "-m", "cornserve.services.sidecar.server"]
