# Scripts for stack testing and operations

Shell scripts for running ingestion, evidence gathering, and the full stack with sample data. Run from **repo root** or from this folder.

| Script | Purpose |
|--------|--------|
| `ingest_otel.sh` | Ingest a sample OTEL trace (default: `sample-data/example_otel_2.json`). Output: `sample-data/output/ingestion_response.json`. |
| `run_evidence.sh` | Call the Evidence Gathering agent (default intent: "What does Miss-Marple do?"). Output: `sample-data/output/evidence_response.json`. |
| `ingest_and_evidence.sh` | Run ingestion then evidence. Uses default sample and default intent. |

**From repo root:**

```bash
./scripts/ingest_otel.sh
./scripts/run_evidence.sh
./scripts/ingest_and_evidence.sh
```

**From scripts/:**

```bash
cd scripts && ./ingest_otel.sh
./run_evidence.sh "What is Miss-Marple responsible for?"
./ingest_and_evidence.sh
```

Requires: `curl`, `jq`. Start the unified app first (e.g. `docker compose up` or `uvicorn gateway.app.main:app --port 9004`).
