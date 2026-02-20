# syntax=docker/dockerfile:1.4
FROM ghcr.io/cisco-eti/sre-python-docker:v3.11.9-hardened-debian-12

RUN apt-get update && apt-get install -y git build-essential cmake && rm -rf /var/lib/apt/lists/*

RUN useradd -u 1001 app
RUN mkdir /home/app/ && chown -R app:app /home/app

WORKDIR /home/app

RUN pip install poetry==1.8.0 --break-system-packages

COPY --chown=app:app pyproject.toml poetry.lock README.md ./

RUN poetry config virtualenvs.create false
# Pass private dependency tokens via BuildKit secrets: --secret id=gh_token,src=/path/to/token and/or npm_token
RUN --mount=type=secret,id=gh_token,required=false \
    --mount=type=secret,id=npm_token,required=false \
    TOKEN="$(cat /run/secrets/gh_token 2>/dev/null || cat /run/secrets/npm_token 2>/dev/null || true)" && \
    if [ -z "$TOKEN" ]; then \
        echo "WARNING: No gh_token or npm_token secret provided; private dependencies may fail to install."; \
    else \
        # legit:ignore-pipeline – private git dependencies require temporary token rewrite
        git config --global url."https://x-access-token:${TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi && \
    poetry install --no-interaction --no-ansi --only main --no-root && \
    if [ -n "$TOKEN" ]; then \
        # legit:ignore-pipeline – removing rewrite configured above
        git config --global --remove-section url."https://x-access-token:${TOKEN}@github.com/" 2>/dev/null || true; \
    fi

COPY --chown=app:app ingestion-cognitive-agent ./ingestion-cognitive-agent

EXPOSE 8086

USER app