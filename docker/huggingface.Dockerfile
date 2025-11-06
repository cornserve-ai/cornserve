FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ADD ./python /workspace/cornserve/python

WORKDIR /workspace/cornserve/python
RUN pip install -e '.[huggingface-te]'

ENTRYPOINT ["python", "-u", "-m", "cornserve.task_executors.huggingface.entrypoint"]
