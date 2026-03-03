#!/usr/bin/env bash
# Call the Evidence Gathering agent (e.g. "What does Miss-Marple do?").
# Usage: ./run_evidence.sh [intent]
# Default intent: "What does Miss-Marple do?"
# Saves response to sample-data/output/evidence_response.json.
# Run from repo root or scripts/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/sample-data/output"
mkdir -p "$OUTPUT_DIR"
INTENT="${1:-What does Miss-Marple do?}"
OUTPUT_FILE="${OUTPUT_DIR}/evidence_response.json"

echo "Evidence Gathering: intent=\"$INTENT\""
echo "POSTing to evidence ..."
BODY=$(jq -n \
  --arg intent "$INTENT" \
  --arg request_id "eg-$(date +%s)" \
  '{ header: { workspace_id: "default", mas_id: "default" }, request_id: $request_id, payload: { intent: $intent } }')
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$OUTPUT_FILE" -X POST "http://localhost:9004/api/knowledge-mgmt/reasoning/evidence" \
  -H "Content-Type: application/json" \
  -d "$BODY")

echo "HTTP $HTTP_CODE"
echo "Response saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE" | jq . 2>/dev/null || cat "$OUTPUT_FILE"
echo ""

if [ "$HTTP_CODE" -ge 400 ]; then
  exit 1
fi
