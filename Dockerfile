# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    NOWCAST_BUNDLE_DIR=/app/artifacts/operational/residual_v1 \
    NOWCAST_CACHE_DIR=/tmp/nowcast-cache \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra data --extra baseline --extra demo --extra operational
COPY src ./src
COPY app/live_nowcast.py ./app/live_nowcast.py
COPY artifacts/operational/residual_v1 ./artifacts/operational/residual_v1
RUN uv sync --frozen --no-dev --extra data --extra baseline --extra demo --extra operational

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD test -f "$NOWCAST_BUNDLE_DIR/manifest.json" && curl -f http://127.0.0.1:7860/_stcore/health || exit 1
CMD ["/app/.venv/bin/streamlit", "run", "app/live_nowcast.py"]
