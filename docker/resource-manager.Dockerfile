FROM python:3.11.11

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e .[resource-manager]

ENTRYPOINT ["python", "-m", "cornserve.services.resource_manager.server"]
