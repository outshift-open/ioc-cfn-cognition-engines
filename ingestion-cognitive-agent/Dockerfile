# ── Stage 1: install dependencies into an isolated venv ─────────────────────
FROM ghcr.io/cisco-eti/sre-python-docker:v3.11.9-hardened-debian-12 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM ghcr.io/cisco-eti/sre-python-docker:v3.11.9-hardened-debian-12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=10001:10001 app/ app/

RUN groupadd -g 10001 app \
    && useradd -r -u 10001 -g app -d /home/app -m app

USER app:app

EXPOSE 8086
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8086"]
