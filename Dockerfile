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
# patchelf
#
# onnxruntime 1.12.1 wheels request an executable stack.
# GitHub-hosted kernels refuse to map that stack, so use patchelf >= 0.18
# and clear the executable-stack requirement from the ORT shared libraries.
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
            echo "Unsupported architecture for patchelf: $arch"; \
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
# Fix ONNX Runtime executable-stack requirement
# ---------------------------------------------------------------------------
RUN set -eux; \
    if [ -d /usr/local/lib/python3.9/site-packages/onnxruntime ]; then \
        find /usr/local/lib/python3.9/site-packages/onnxruntime \
            -name "*.so" \
            -exec patchelf --clear-execstack {} \; \
    fi

# ---------------------------------------------------------------------------
# Validate native Python packages independently.
#
# Keeping these as separate RUN commands makes an ARM64 native-library
# failure immediately identifiable instead of hiding it inside one import.
# ---------------------------------------------------------------------------
RUN python -c "import yaml; print('PyYAML OK')"

RUN python -c \
    "import cv2; print('OpenCV OK:', cv2.__version__)"

RUN python -c \
    "import onnxruntime; print('ONNX Runtime OK:', onnxruntime.__version__)"

RUN python -c \
    "import insightface; print('InsightFace OK')"

RUN python -c \
    "import paddle; print('Paddle OK:', paddle.__version__)"

RUN python -c \
    "import paddleocr; print('PaddleOCR OK')"

RUN python -c \
    "import mediapipe; print('MediaPipe OK')"

# ---------------------------------------------------------------------------
# Application source
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
# Model files
#
# IMPORTANT:
# Do not execute the "--all" model download during an emulated cross-platform
# build. Native ML libraries can segfault under QEMU even when Python itself
# works correctly.
#
# TARGETARCH and BUILDARCH are supplied by BuildKit/buildx.
# ---------------------------------------------------------------------------
ARG TARGETARCH
ARG BUILDARCH

RUN set -eux; \
    echo "Build architecture : ${BUILDARCH}"; \
    echo "Target architecture: ${TARGETARCH}"; \
    mkdir -p /app/models /app/logs /app/temp

# ---------------------------------------------------------------------------
# YuNet
#
# This is a static ONNX model and can safely be downloaded on either
# architecture.
# ---------------------------------------------------------------------------
RUN set -eux; \
    if [ ! -f /app/models/yunet.onnx ]; then \
        wget -q \
            --show-progress \
            -O /app/models/yunet.onnx \
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"; \
    fi; \
    test -s /app/models/yunet.onnx

# ---------------------------------------------------------------------------
# Application-specific model downloads
#
# Run the normal downloader, but NEVER invoke "--all" when the target is
# being cross-built/emulated.
#
# If BUILDARCH == TARGETARCH, native execution is available and --all can
# safely be attempted.
# ---------------------------------------------------------------------------
RUN set -eux; \
    python scripts/download_models.py; \
    if [ "${TARGETARCH}" = "${BUILDARCH}" ]; then \
        echo "Native build detected (${TARGETARCH}); downloading all models."; \
        python scripts/download_models.py --all; \
    else \
        echo "Cross-platform build detected (${BUILDARCH} -> ${TARGETARCH});"; \
        echo "skipping native --all model download."; \
    fi

# ---------------------------------------------------------------------------
# Final model directory sanity check
# ---------------------------------------------------------------------------
RUN set -eux; \
    echo "=== Models ==="; \
    find /app/models -maxdepth 2 -type f -print | sort; \
    echo "=== Model directory size ==="; \
    du -sh /app/models

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
