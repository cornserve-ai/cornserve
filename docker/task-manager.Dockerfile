FROM python:3.11.11

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e .[task-manager]

ENTRYPOINT ["python", "-m", "cornserve.services.task_manager.server"]
