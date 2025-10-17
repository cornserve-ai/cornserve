FROM ubuntu:24.04

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH="/root/.local/bin:$PATH"
RUN uv venv --python 3.11 --seed /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ADD ./python /workspace/cornserve/python

WORKDIR /workspace/cornserve/python
RUN uv pip install -e .[task-dispatcher] && uv cache clean

ENTRYPOINT ["python", "-u", "-m", "cornserve.services.task_dispatcher.entrypoint"]
