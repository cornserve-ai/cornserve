FROM python:3.11.11

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e .[gateway]

ENTRYPOINT ["python", "-m", "cornserve.services.gateway.entrypoint"]
