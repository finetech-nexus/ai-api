# Same base as ./kyc/Dockerfile.ml-backend
# Python 3.9, PaddleOCR, MediaPipe.
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
# patchelf >= 0.18
# ---------------------------------------------------------------------------
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
        x86_64) \
            patchelf_arch="x86_64" \
            ;; \
        aarch64) \
            patchelf_arch="aarch64" \
            ;; \
        *) \
            echo "Unsupported architecture: ${arch}"; \
            exit 1 \
            ;; \
    esac; \
    wget -qO /tmp/patchelf.tgz \
      "https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-${patchelf_arch}.tar.gz"; \
    tar -C /usr/local -xzf /tmp/patchelf.tgz; \
    rm -f /tmp/patchelf.tgz; \
    patchelf --version

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
# ONNX Runtime executable-stack workaround
# ---------------------------------------------------------------------------
RUN set -eux; \
    ORT_DIR="/usr/local/lib/python3.9/site-packages/onnxruntime"; \
    if [ -d "$ORT_DIR" ]; then \
        find "$ORT_DIR" \
            -name "*.so" \
            -exec patchelf --clear-execstack {} \; \
    fi

# ---------------------------------------------------------------------------
# Copy application
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
# Build architecture information
# ---------------------------------------------------------------------------
ARG TARGETARCH
ARG BUILDARCH

RUN echo "BUILDARCH=${BUILDARCH}" && \
    echo "TARGETARCH=${TARGETARCH}" && \
    uname -m && \
    python -c "import platform; print('Python architecture:', platform.machine())"

# ---------------------------------------------------------------------------
# Model directories
# ---------------------------------------------------------------------------
RUN mkdir -p \
    /app/models \
    /app/insightface/models/buffalo_l \
    /app/logs \
    /app/temp

# ---------------------------------------------------------------------------
# Download static model weights.
#
# IMPORTANT:
# This does NOT initialize PaddleOCR, MediaPipe, InsightFace, etc.
# Therefore ARM64 builds don't execute the native ML stack under BuildKit/QEMU.
# ---------------------------------------------------------------------------
RUN set -eux; \
    python scripts/download_models.py

# ---------------------------------------------------------------------------
# Native model warm-up
#
# Only perform this when the build architecture and target architecture match.
#
# ARM64:
#   Download weights only.
#   Do NOT execute the native ML runtimes during docker build.
#
# AMD64 native:
#   Warm all models and fail the image build if one cannot initialize.
# ---------------------------------------------------------------------------
RUN set -eux; \
    if [ "${TARGETARCH}" = "${BUILDARCH}" ] && [ "${TARGETARCH}" = "amd64" ]; then \
        echo "Native AMD64 build: warming all ML models"; \
        python scripts/download_models.py --warm; \
    else \
        echo "Skipping ML warm-up for ${BUILDARCH} -> ${TARGETARCH}"; \
        echo "Models were downloaded successfully."; \
    fi

# ---------------------------------------------------------------------------
# Verify model files exist
# ---------------------------------------------------------------------------
RUN set -eux; \
    test -s /app/models/yunet.onnx; \
    test -s /app/insightface/models/buffalo_l/det_10g.onnx; \
    test -s /app/insightface/models/buffalo_l/w600k_r50.onnx; \
    echo "Required model files:"; \
    find /app/models /app/insightface/models -type f -print | sort

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
