FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ADD ./python /workspace/cornserve/python

WORKDIR /workspace/cornserve/python
RUN apt-get update \
      && apt-get install -y --no-install-recommends curl \
      && rm -rf /var/lib/apt/lists/* \
      && curl -LO https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp311-cp311-linux_x86_64.whl \
      && pip install flash_attn-2.8.1+cu128torch2.9-cp311-cp311-linux_x86_64.whl \
      && rm flash_attn-2.8.1+cu128torch2.9-cp311-cp311-linux_x86_64.whl
RUN pip install -e '.[huggingface-te]'

ENTRYPOINT ["python", "-u", "-m", "cornserve.task_executors.huggingface.entrypoint"]
