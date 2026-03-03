#!/usr/bin/env bash
# Ingest sample OTEL trace via POST (used for stack testing).
# Usage: ./ingest_otel.sh [path-to-otel.json]
# Default: sample-data/example_otel_2.json (fallback: example_otel_2.json at repo root).
# Run from repo root or scripts/; outputs to sample-data/output/ingestion_response.json.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/sample-data/output"
mkdir -p "$OUTPUT_DIR"
DEFAULT_OTEL="${REPO_ROOT}/sample-data/example_otel_2.json"
[[ -f "$DEFAULT_OTEL" ]] || DEFAULT_OTEL="${REPO_ROOT}/example_otel_2.json"
OTEL_FILE="${1:-$DEFAULT_OTEL}"
PAYLOAD_FILE="${SCRIPT_DIR}/.ingest_payload.json"
INGESTION_OUTPUT="${OUTPUT_DIR}/ingestion_response.json"

echo "Building payload from $OTEL_FILE ..."
jq -n \
  --slurpfile data "$OTEL_FILE" \
  '{ header: { workspace_id: "default", mas_id: "default" }, request_id: "ingest-otel2", payload: { metadata: { format: "observe-sdk-otel" }, data: $data[0] } }' \
  > "$PAYLOAD_FILE"

echo "POSTing to ingestion ..."
RESPONSE="$(mktemp)"
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$RESPONSE" -X POST "http://localhost:9004/api/knowledge-mgmt/extraction" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_FILE")

cp "$RESPONSE" "$INGESTION_OUTPUT"
echo "HTTP $HTTP_CODE"
echo "Response saved to $INGESTION_OUTPUT"
cat "$RESPONSE" | jq . 2>/dev/null || cat "$RESPONSE"
echo ""
rm -f "$RESPONSE"
echo "Cleaning up payload file."
rm -f "$PAYLOAD_FILE"

if [ "$HTTP_CODE" -ge 400 ]; then
  exit 1
fi
