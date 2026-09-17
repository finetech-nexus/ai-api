# Use Python 3.9 — InsightFace / MediaPipe / PaddleOCR compatible
FROM python:3.9-slim

WORKDIR /app

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
    && rm -rf /var/lib/apt/lists/*

# Debian's patchelf is <0.18 and has no --clear-execstack. onnxruntime 1.12.1
# wheels request an executable stack; GitHub-hosted kernels refuse to map that.
RUN set -eux; \
    arch="$(uname -m)"; \
    wget -qO /tmp/patchelf.tgz \
      "https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-${arch}.tar.gz"; \
    tar -C /usr/local -xzf /tmp/patchelf.tgz; \
    rm /tmp/patchelf.tgz; \
    patchelf --version

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --timeout=300 -r requirements.txt

RUN find /usr/local/lib/python3.9/site-packages/onnxruntime -name "*.so" \
        -exec patchelf --clear-execstack {} \; \
    && python -c "import yaml, cv2, insightface, onnxruntime, paddleocr, mediapipe"

COPY main.py ./
COPY api ./api
COPY core ./core
COPY domains ./domains
COPY vendor ./vendor
COPY scripts ./scripts

ENV PYTHONPATH=/app:/app/vendor/kyc
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Provision the InsightFace pack from this repo's public weights release, then
# construct every model once: startup needs no network afterwards, and a model
# that cannot load fails the build instead of leaving the pod unready.
RUN mkdir -p /app/vendor/kyc/logs /app/vendor/kyc/temp \
    && python scripts/download_models.py --all

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
