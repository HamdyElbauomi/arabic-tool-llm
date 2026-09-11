# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.6 wheels retain support for the project's Pascal GPU.
# The wheel dependencies provide CUDA libraries; no host CUDA toolkit is copied.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt \
    && python -m pip check

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/logs /app/models/qwen3-0.6b-tool-calling-v2-lora /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/tools.json ./configs/tools.json

# Optional: existing tests use fake agents and need no GPU or model download.
FROM base AS test
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pytest==9.1.1 httpx==0.28.1
COPY --chown=app:app tests/ ./tests/
USER app
RUN python -m pytest tests -v -p no:cacheprovider

FROM base AS runtime
USER app
EXPOSE 8000

# Startup loads Qwen + LoRA before the API can serve requests.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD python -c "import json, urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); assert r.get('status') == 'healthy' and r.get('model_loaded') is True"

# One worker means one model copy on the 4 GB GPU.
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
