FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN pip install 'cornserve[sidecar-api, dev-no-gpu]'

ADD . /workspace

WORKDIR /workspace

CMD ["bash"]
