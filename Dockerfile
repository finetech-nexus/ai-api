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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade \
        pip \
        setuptools \
        wheel \
    && pip install \
        --no-cache-dir \
        --timeout=300 \
        -r requirements.txt

RUN python -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
RUN python -c "import cv2; print('OpenCV:', cv2.__version__)"
RUN python -c "import insightface; print('InsightFace OK')"
RUN python -c "import paddle; print('Paddle:', paddle.__version__)"
RUN python -c "import paddleocr; print('PaddleOCR OK')"
RUN python -c "import mediapipe; print('MediaPipe OK')"

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

ARG TARGETARCH
ARG BUILDARCH

RUN echo "BUILDARCH=${BUILDARCH}" && \
    echo "TARGETARCH=${TARGETARCH}" && \
    uname -m

RUN mkdir -p \
    /app/models \
    /app/insightface/models/buffalo_l \
    /app/logs \
    /app/temp

# Download weights only. No native ML initialization.
RUN python scripts/download_models.py

# Only warm models on native AMD64.
RUN if [ "${TARGETARCH}" = "${BUILDARCH}" ] && [ "${TARGETARCH}" = "amd64" ]; then \
        echo "Native AMD64: warming models"; \
        python scripts/download_models.py --warm; \
    else \
        echo "Skipping build-time ML warm-up for ${BUILDARCH} -> ${TARGETARCH}"; \
    fi

RUN test -s /app/models/yunet.onnx && \
    test -s /app/insightface/models/buffalo_l/det_10g.onnx && \
    test -s /app/insightface/models/buffalo_l/w600k_r50.onnx

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=90s \
    --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "python -m uvicorn api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
