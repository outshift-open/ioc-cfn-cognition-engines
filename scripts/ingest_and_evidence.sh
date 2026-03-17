#!/usr/bin/env bash
# Run ingestion (extract concepts) then Evidence Gathering against the unified gateway.
# Usage: ./ingest_and_evidence.sh [extraction_body.json] [evidence_body.json]
#   extraction_body.json : default tests/extraction_request_body.json
#   evidence_body.json   : default tests/evidence_request_body.json
# Example: ./ingest_and_evidence.sh
# Example: ./ingest_and_evidence.sh tests/extraction_request_body.json tests/evidence_request_body.json
# Run from repo root or scripts/; outputs under sample-data/output/.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_URL="${BASE_URL:-http://localhost:9004}"
EXTRACTION_BODY="${1:-tests/data/extraction_request_body.json}"
EVIDENCE_BODY="${2:-tests/data/evidence_request_body.json}"

mkdir -p sample-data/output

echo "========== 1/2 Ingestion =========="
echo "POST ${BASE_URL}/api/knowledge-mgmt/extraction"
echo "Body: ${EXTRACTION_BODY}"
HTTP_CODE=$(curl -s -w "%{http_code}" -o sample-data/output/ingestion_response.json -X POST \
  "${BASE_URL}/api/knowledge-mgmt/extraction" \
  -H "Content-Type: application/json" \
  -d @"$EXTRACTION_BODY")
echo "HTTP ${HTTP_CODE}"
echo "Response saved to sample-data/output/ingestion_response.json"
jq . sample-data/output/ingestion_response.json 2>/dev/null || cat sample-data/output/ingestion_response.json

if [ "$HTTP_CODE" -ge 400 ]; then
  echo "Ingestion failed (HTTP ${HTTP_CODE}). Aborting."
  exit 1
fi

echo ""
echo "========== 2/2 Evidence Gathering =========="
echo "POST ${BASE_URL}/api/knowledge-mgmt/reasoning/evidence"
echo "Body: ${EVIDENCE_BODY}"
HTTP_CODE=$(curl -s -w "%{http_code}" -o sample-data/output/evidence_response.json -X POST \
  "${BASE_URL}/api/knowledge-mgmt/reasoning/evidence" \
  -H "Content-Type: application/json" \
  -d @"$EVIDENCE_BODY")
echo "HTTP ${HTTP_CODE}"
echo "Response saved to sample-data/output/evidence_response.json"
jq . sample-data/output/evidence_response.json 2>/dev/null || cat sample-data/output/evidence_response.json

if [ "$HTTP_CODE" -ge 400 ]; then
  echo "Evidence gathering failed (HTTP ${HTTP_CODE})."
  exit 1
fi

echo ""
echo "Done. Outputs saved under sample-data/output/: ingestion_response.json, evidence_response.json"
