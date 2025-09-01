# HuggingFace Task Executor for Qwen-Image and Qwen 2.5 Omni
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers>=4.52.3 \
    diffusers>=0.35.0 \
    qwen-omni-utils[decord] \
    fastapi \
    uvicorn[standard] \
    soundfile \
    pillow \
    numpy \
    tyro \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-threading \
    aiohttp \
    rich

# Copy the Cornserve package
COPY python/ /app/python/

# Install Cornserve package
RUN cd python && pip install -e .

# Set up environment
ENV PYTHONPATH=/app/python
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV DIFFUSERS_CACHE=/root/.cache/huggingface/diffusers

# Create cache directories
RUN mkdir -p /root/.cache/huggingface/transformers && \
    mkdir -p /root/.cache/huggingface/diffusers

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "-m", "cornserve.task_executors.huggingface.entrypoint"]