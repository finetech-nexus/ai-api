# Same base as ./kyc/Dockerfile.ml-backend (Python 3.9, PaddleOCR, MediaPipe).
FROM python:3.9-slim

WORKDIR /app

# System deps from kyc, plus wget for model fetch during the image build.
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

# Same idea as kyc/Dockerfile.ml-backend: fetch YuNet during the image build.
# InsightFace's recognition weights exceed GitHub's 100 MB git limit, so they
# come from this repo's public weights-v1 release (see scripts/download_models.py).
ARG TARGETARCH
ARG BUILDARCH
RUN mkdir -p /app/models /app/logs /app/temp \
    && if [ ! -f /app/models/yunet.onnx ]; then \
         wget -O /app/models/yunet.onnx \
           https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx; \
       fi \
    && python scripts/download_models.py \
    && if [ "$TARGETARCH" = "$BUILDARCH" ]; then \
         python scripts/download_models.py --all; \
       else \
         echo "Skipping --all: emulated ${TARGETARCH} on ${BUILDARCH}"; \
       fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "python -m uvicorn api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
