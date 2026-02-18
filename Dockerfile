# Qwen3-TTS OpenAI-compatible Server — NVIDIA CUDA
#
# Build:  docker build -t qwen3-tts .
# Run:    docker run --gpus all -p 8880:8880 -v ~/qwen3-tts:/root/qwen3-tts qwen3-tts

# =============================================================================
# Stage 1: Base image with system dependencies
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE} AS base

WORKDIR /opt/qwen3-tts

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.11 \
  python3.11-venv \
  python3.11-dev \
  python3-pip \
  build-essential \
  git \
  curl \
  ffmpeg \
  libsox-dev \
  ninja-build \
  libsndfile1 \
  sox \
  && rm -rf /var/lib/apt/lists/* \
  && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
  && ln -sf /usr/bin/python3 /usr/bin/python
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
  torch==2.9.1 \
  torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu129 && \
  pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.11/flash_attn-2.8.3%2Bcu129torch2.9-cp311-cp311-linux_x86_64.whl


COPY pyproject.toml ./

# Install Python dependencies

# Install the main package dependencies

# Copy application code
COPY requirements.txt .

# Install the package (makes qwen_tts importable)
RUN pip install --no-cache-dir -e .

COPY . .
# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
  && mkdir -p /tmp/numba_cache \
  && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

RUN mkdir -p /home/appuser/qwen3-tts/voice_library
COPY config.yaml /home/appuser/qwen3-tts/config.yaml
# Default config location
RUN mkdir -p /root/qwen3-tts/voice_library
COPY config.yaml /root/qwen3-tts/config.yaml

ENV TTS_BACKEND=optimized \
    HOST=0.0.0.0 \
    PORT=8880 \
    WORKERS=1

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "api.main"]
