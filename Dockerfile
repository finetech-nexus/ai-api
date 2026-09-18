FROM python:3.9-slim

WORKDIR /app

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    gcc \
    g++ \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade \
        pip \
        setuptools \
        wheel \
    && pip install \
        --no-cache-dir \
        --timeout=300 \
        -r requirements.txt

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
COPY api ./api
COPY aml ./aml
COPY app ./app
COPY configs ./configs
COPY utils ./utils
COPY models ./models
COPY scripts ./scripts

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# ---------------------------------------------------------------------------
# Build information
# ---------------------------------------------------------------------------
ARG TARGETARCH
ARG BUILDARCH

RUN echo "Build architecture : ${BUILDARCH}" && \
    echo "Target architecture: ${TARGETARCH}" && \
    uname -m

# ---------------------------------------------------------------------------
# Model directories
# ---------------------------------------------------------------------------
RUN mkdir -p \
    /app/models \
    /app/insightface/models/buffalo_l \
    /app/logs \
    /app/temp

# ---------------------------------------------------------------------------
# Download model weights ONLY.
#
# download_models.py must NOT be called with --all here.
# --all initializes PaddleOCR / InsightFace / MediaPipe and can segfault
# inside BuildKit/QEMU on ARM64.
# ---------------------------------------------------------------------------
RUN python scripts/download_models.py

# ---------------------------------------------------------------------------
# Verify that static model files were downloaded.
# No ML libraries are imported here.
# ---------------------------------------------------------------------------
RUN set -eux; \
    test -s /app/models/yunet.onnx; \
    test -s /app/insightface/models/buffalo_l/det_10g.onnx; \
    test -s /app/insightface/models/buffalo_l/w600k_r50.onnx

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=90s \
    --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "python -m uvicorn api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
