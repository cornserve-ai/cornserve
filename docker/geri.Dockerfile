FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ADD ./python /workspace/cornserve/python

WORKDIR /workspace/cornserve/python
RUN pip install -e '.[geri]'

ENTRYPOINT ["python", "-u", "-m", "cornserve.task_executors.geri.entrypoint"]
