# TODO: Use multi-stage build to reduce image size.
#       The `devel` image was used because we need `nvcc` to install flash-attn.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e '.[eric]'

ENTRYPOINT ["python", "-u", "-m", "cornserve.task_executors.eric.entrypoint"]
