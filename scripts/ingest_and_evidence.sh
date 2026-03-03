#!/usr/bin/env bash
# Run ingestion (OTEL -> FAISS) then Evidence Gathering (e.g. "What does Miss-Marple do?").
# Usage: ./ingest_and_evidence.sh [path-to-otel.json] [intent]
#   path-to-otel.json: default sample-data/example_otel_2.json
#   intent: default "What does Miss-Marple do?"
# Example: ./ingest_and_evidence.sh
# Example: ./ingest_and_evidence.sh ../sample-data/example_otel_2.json "What is Miss-Marple responsible for?"
# Run from repo root or scripts/; outputs under sample-data/output/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_OTEL="${REPO_ROOT}/sample-data/example_otel_2.json"
[[ -f "$DEFAULT_OTEL" ]] || DEFAULT_OTEL="${REPO_ROOT}/example_otel_2.json"
OTEL_FILE="${1:-$DEFAULT_OTEL}"
INTENT="${2:-What does Miss-Marple do?}"

echo "========== 1/2 Ingestion =========="
"$SCRIPT_DIR/ingest_otel.sh" "$OTEL_FILE" || exit 1

echo ""
echo "========== 2/2 Evidence Gathering =========="
"$SCRIPT_DIR/run_evidence.sh" "$INTENT" || exit 1

echo ""
echo "Done. Outputs saved under sample-data/output/: ingestion_response.json, evidence_response.json"
