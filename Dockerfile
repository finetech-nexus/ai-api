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
    patchelf \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --timeout=300 -r requirements.txt \
    && find /usr/local/lib/python3.9/site-packages/onnxruntime -name "*.so" -exec patchelf --clear-execstack {} \; 2>/dev/null || true

COPY main.py ./
COPY api ./api
COPY core ./core
COPY domains ./domains
COPY vendor ./vendor
COPY scripts ./scripts

ENV PYTHONPATH=/app:/app/vendor/kyc
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

RUN mkdir -p /app/vendor/kyc/models /app/vendor/kyc/logs /app/vendor/kyc/temp \
    && wget -O /app/vendor/kyc/models/yunet.onnx \
       https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
