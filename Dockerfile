# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    MEMORYBOT_DB_PATH=/data/chroma_db \
    MEMORYBOT_EVAL_LOG=/data/evaluation/eval_log.json \
    MEMORYBOT_TIMEZONE=Asia/Singapore
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
COPY requirements.txt ./
# CPU wheels avoid shipping CUDA libraries. Both x86_64 and ARM64 are supported.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && python -m pip install -r requirements.txt \
    && python -m pip check
COPY --chown=app:app Backend ./Backend
COPY --chown=app:app Chatbot ./Chatbot
COPY --chown=app:app Evaluation ./Evaluation
COPY --chown=app:app Memory ./Memory
COPY --chown=app:app RAG ./RAG
COPY --chown=app:app Tools ./Tools
COPY --chown=app:app Frontend ./Frontend
RUN mkdir -p /data/chroma_db /data/evaluation /home/app/.cache/huggingface \
    && chown -R app:app /data /home/app/.cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
# Sessions are in process memory, so use one worker.
CMD ["python", "-m", "uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
