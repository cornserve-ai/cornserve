FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ADD . /workspace

WORKDIR /workspace
RUN pip install 'cornserve[sidecar-api, dev-no-gpu]'

CMD ["bash"]
