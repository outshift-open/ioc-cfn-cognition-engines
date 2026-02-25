# ── Stage 1: install all dependencies into an isolated venv ─────────────────
# NOTE: swap python:3.11-slim → ghcr.io/cisco-eti/sre-python-docker:v3.11.9-hardened-debian-12
#       once you are authenticated to ghcr.io (docker login ghcr.io)
FROM ghcr.io/cisco-eti/sre-python-docker:v3.11.9-hardened-debian-12 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
# Install Poetry (no venv needed for Poetry itself in the builder)
RUN pip install --no-cache-dir "poetry>=1.8.0" "poetry-plugin-export" && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project false && \
    poetry config virtualenvs.path /opt/venv-store

# Copy only the dependency manifest first (better layer caching)
COPY pyproject.toml ./
# poetry.lock is optional – copy if present
COPY poetry.loc[k] ./

# Resolve/generate the lock file, export to plain requirements, then install.
# Poetry's [[source]] block injects the pytorch-cpu extra index automatically.
RUN poetry lock --no-update 2>/dev/null || poetry lock
RUN poetry export \
    --without dev \
    --without-hashes \
    --format requirements.txt \
    -o /tmp/requirements-export.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-export.txt

# Aggressive cleanup in a separate step so failures above are never masked
RUN find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'tests'    -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'test'     -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -name '*.pyc' -delete 2>/dev/null || true \
    && find /opt/venv -name '*.pyo' -delete 2>/dev/null || true

# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM ghcr.io/cisco-eti/sre-python-docker:v3.11.9-hardened-debian-12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy each service into its own subdirectory
COPY --chown=1000:1000 gateway/app/            /app/gateway/app/
COPY --chown=1000:1000 ingestion-cognitive-agent/app/  /app/ingestion/app/
COPY --chown=1000:1000 evidence-gathering-agent/app/   /app/evidence/app/
COPY --chown=1000:1000 caching-layer/app/              /app/caching/app/

# Copy supervisor config
COPY --chown=1000:1000 supervisord.conf /etc/supervisord.conf

RUN adduser --disabled-password --gecos "" --uid 1000 app 2>/dev/null || true

USER app

# Gateway is the only externally exposed port; others are internal
EXPOSE 8000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
