FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY app/ app/
# Create non-root user and take ownership of workdir
RUN groupadd -g 10001 app && \
    useradd -r -u 10001 -g app -d /home/app -m app && \
    chown -R app:app /app /home/app
USER app:app

EXPOSE 8087
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8087"]

