# syntax=docker/dockerfile:1.7
# Lean CPU inference image: FastAPI + ONNX Runtime + numpy/OpenCV-headless. No torch, no Ultralytics.
# Build from the repo root:  docker build -f docker/Dockerfile -t brain-tumor-detection .
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /src
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Hash-locked dependencies first (better layer caching + supply-chain integrity), then the package itself.
COPY requirements/serve-cpu.txt ./requirements.txt
RUN pip install --require-hashes --no-deps -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="brain-tumor-detection" \
      org.opencontainers.image.description="Brain tumour detection & segmentation API (YOLO26-seg, quantised ONNX, CPU)" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/leon2378/brain-tumor-detection"
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BTD_MODEL_PATH=/app/models/model.onnx \
    BTD_PROVIDERS=cpu \
    BTD_LOG_JSON=true \
    BTD_HOST=0.0.0.0 \
    BTD_PORT=8000
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# models/ holds model.onnx + model.json (from `btd package`); an empty folder yields a not-ready service.
COPY --chown=app:app models/ /app/models/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import os, sys, urllib.request; url = f'http://127.0.0.1:{os.environ.get(\"BTD_PORT\", \"8000\")}/health'; sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"]
# `docker run IMAGE` serves on BTD_PORT. SageMaker runs `docker run IMAGE serve` and sets BTD_PORT=8080.
# Full path: SageMaker Serverless, like Lambda, replaces PATH, and a bare "btd" then can't even start.
ENTRYPOINT ["/opt/venv/bin/btd"]
CMD ["serve"]
