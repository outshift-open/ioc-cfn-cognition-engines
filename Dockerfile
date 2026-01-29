FROM ghcr.io/cisco-eti/sre-python-docker:v3.11.9-hardened-debian-12

# Install git as root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Add user app
RUN useradd -u 1001 app
RUN mkdir /home/app/ && chown -R app:app /home/app

WORKDIR /home/app

# Install Poetry as root
RUN pip install poetry==1.8.0 --break-system-packages

# Copy Poetry configuration files
COPY --chown=app:app pyproject.toml poetry.lock ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Accept tokens
ARG GITHUB_TOKEN
ARG NPM_TOKEN

# legit-security-ignore: External resource with token is required for private git dependencies
# Use credential helper - more secure than URL rewriting
RUN TOKEN="${GITHUB_TOKEN:-$NPM_TOKEN}" && \
    if [ -z "$TOKEN" ]; then \
        echo "ERROR: Neither GITHUB_TOKEN nor NPM_TOKEN is set!"; \
        exit 1; \
    fi && \
    git config --global credential.helper store && \
    mkdir -p /root && \
    echo "https://x-access-token:${TOKEN}@github.com" > /root/.git-credentials && \
    chmod 600 /root/.git-credentials && \
    poetry install --no-interaction --no-ansi --only main --no-root && \
    rm -f /root/.git-credentials && \
    git config --global --unset credential.helper

# Copy application code
COPY --chown=app:app app/src/ .

# Expose both ports
EXPOSE 8086

# Switch to app user
USER app