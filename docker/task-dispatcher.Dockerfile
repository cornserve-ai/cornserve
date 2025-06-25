FROM python:3.11.11

ADD ./python /workspace/cornserve/python

WORKDIR /workspace/cornserve/python
RUN pip install -e .[task-dispatcher]

ENTRYPOINT ["python", "-m", "cornserve.services.task_dispatcher.entrypoint"]
