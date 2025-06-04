# Build flash-attn wheel inside the `devel` image which has `nvcc`.
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel AS builder

ARG max_jobs=64
ENV MAX_JOBS=${max_jobs}
ENV NVCC_THREADS=8
RUN pip wheel -w /tmp/wheels --no-build-isolation --no-deps --verbose flash-attn

# Actual Eric runs inside the `runtime` image. Just copy over the flash-attn wheel.
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime AS eric

COPY --from=builder /tmp/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e '.[eric]'

ENTRYPOINT ["python", "-u", "-m", "cornserve.task_executors.eric.entrypoint"]

# Eric that has audio support.
FROM eric AS eric-audio

RUN pip install -e '.[audio]'
