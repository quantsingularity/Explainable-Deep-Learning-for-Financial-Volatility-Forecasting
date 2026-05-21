# Multi-stage Dockerfile for Volatility Forecasting System
# Supports both GPU and CPU variants

ARG VARIANT=cpu
FROM tensorflow/tensorflow:2.14.0${VARIANT:+-gpu} as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY code/requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary runtime directories
RUN mkdir -p code/data models docs/figures docs/tables logs mlruns checkpoints

# Set permissions on scripts
RUN chmod +x scripts/run_all.sh scripts/setup.sh scripts/lint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; print(tf.__version__)" || exit 1

# Default — run from code/ so all relative paths resolve correctly
CMD ["bash", "-c", "cd /app/code && python main_pipeline.py"]

# --- Development Stage ---
FROM base as development
RUN pip install ipython ipdb pytest-xdist
EXPOSE 8888 6006
CMD ["bash", "-c", "cd /app/code && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]

# --- Training Stage ---
FROM base as training
ENV MODE=training
CMD ["bash", "-c", "cd /app/code && python -m training.train_multi_horizon"]

# --- Inference/API Stage ---
FROM base as api
EXPOSE 8000
CMD ["bash", "-c", "cd /app/code && python -m serving.api_server"]
