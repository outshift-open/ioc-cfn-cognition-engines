# ── Stage 1: install all dependencies into an isolated venv ─────────────────
FROM python:3.11-slim AS builder

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
# fastembed (ONNX-based) replaces torch+sentence-transformers – no extra index needed.
RUN poetry lock --no-update 2>/dev/null || poetry lock
RUN poetry export \
    --without dev \
    --without-hashes \
    --format requirements.txt \
    -o /tmp/requirements-export.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-export.txt

# Aggressive cleanup in a separate step so failures above are never masked
# Also strip packages that are transitive-only and not needed at inference time:
#   sympy     – onnxruntime shape-inference tooling, not used at runtime
#   hf_xet    – huggingface fast-download protocol, not needed post-install
#   pip       – not needed inside the runtime venv
#   setuptools – same
RUN find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'tests'    -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'test'     -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -name '*.pyc' -delete 2>/dev/null || true \
    && find /opt/venv -name '*.pyo' -delete 2>/dev/null || true \
    && /opt/venv/bin/pip uninstall -y sympy hf_xet pip setuptools 2>/dev/null || true \
    && find /opt/venv -type d -name 'sympy'         -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'hf_xet'        -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name '*.dist-info' -name 'sympy*'   -exec rm -rf {} + 2>/dev/null || true

# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/tmp/fastembed_cache

# Install curl for Docker healthcheck (runtime only)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Pre-download fastembed model using Python (ensures correct directory structure)
# Disable SSL verification for environments with certificate issues
RUN mkdir -p /tmp/fastembed_cache && \
    /opt/venv/bin/python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/tmp/fastembed_cache')"

# Copy each service into its own subdirectory (unified app runs from /app with PYTHONPATH=/app)
COPY --chown=1000:1000 gateway/app/            /app/gateway/app/
COPY --chown=1000:1000 ingestion-cognitive-agent/app/  /app/ingestion/app/
COPY --chown=1000:1000 evidence-gathering-agent/app/   /app/evidence/app/
COPY --chown=1000:1000 caching-layer/app/              /app/caching/app/

RUN adduser --disabled-password --gecos "" --uid 1000 app 2>/dev/null || true \
    && chown -R 1000:1000 /tmp/fastembed_cache 2>/dev/null || true

USER app

# Single process: unified app mounts ingestion and evidence; one port (9004 per Confluence)
ENV PYTHONPATH=/app
EXPOSE 9004
WORKDIR /app
CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "9004"]
