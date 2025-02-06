FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e .

ENTRYPOINT ["python", "-u", "-m", "cornserve.services.sidecar.server"]
