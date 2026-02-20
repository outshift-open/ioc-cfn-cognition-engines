#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8091}
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="${ROOT_DIR}/.demo_uvicorn.log"
PYTHON_BIN=${PYTHON_BIN:-python3}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is required to pretty-print responses." >&2
  exit 1
fi

poetry run uvicorn app.main:app --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "Starting caching-layer service (pid=${SERVER_PID})..."

for _ in {1..30}; do
  if curl -sf "${BASE_URL}/health" >/dev/null; then
    echo "Service is up."
    break
  fi
  sleep 1
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "Server exited early. Check ${LOG_FILE} for details." >&2
    exit 1
  fi
done

if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
  echo "Failed to start service." >&2
  exit 1
fi

store_text() {
  local text="$1"
  echo "Storing: ${text}"
  curl -s -X POST "${BASE_URL}/api/v1/cache/store" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"${text}\"}" | "${PYTHON_BIN}" -m json.tool
}

store_text "The caching layer stores OpenTelemetry summaries."
store_text "Vector search lets agents recall cached cognition payloads quickly."
store_text "Evidence gathered from signals should be searchable later."

printf "\nSearching for 'vector search'...\n"
curl -s -X POST "${BASE_URL}/api/v1/cache/search" \
  -H "Content-Type: application/json" \
  -d '{"text": "vector search", "k": 2}' | "${PYTHON_BIN}" -m json.tool

printf "\nLogs stored at %s\n" "${LOG_FILE}"
