# syntax=docker/dockerfile:1

# ── builder ────────────────────────────────────────────────────────────────
# Resolves the environment from the pinned lock, so the image matches what CI
# tested. Installed editable, because the package resolves its data/models/dashboard
# paths relative to the repo root (config.ROOT); a non-editable install would point
# them into site-packages instead.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependency layer first: it only re-runs when the lock or metadata changes.
# LICENSE is needed because pyproject references it in the license field.
COPY pyproject.toml requirements.lock README.md LICENSE ./
COPY src/ ./src/
RUN pip install -c requirements.lock -e ".[dashboard,api]"

# xgboost's wheel bundles ~400 MB of NVIDIA CUDA libraries for GPU training. This is a
# CPU-only image (and the delivered model is a logistic regression), so drop them. The
# build's own `injury-risk train` step imports xgboost via candidates.py and would fail
# here if removing them broke the import — it does not.
RUN rm -rf /opt/venv/lib/python3.12/site-packages/nvidia \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} +

# ── runtime ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Patch OS packages, then install curl (only needed for the healthcheck).
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app . /app

WORKDIR /app
USER app

# Bake a model in, so `docker run` works out of the box: the dashboard shows real
# predictions and the API answers, with no external artefact to fetch. Training the
# delivered model (default hyperparameters, calibrated) is deterministic and ~15s.
# A deployment that prefers a versioned model can mount over /app/models at runtime.
RUN injury-risk data && injury-risk train --track synthetic

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

# Default to the dashboard — the demo-facing surface. `injury-risk serve` runs the
# API instead (docker-compose runs both).
ENTRYPOINT ["injury-risk"]
CMD ["dashboard", "--port", "8501"]
