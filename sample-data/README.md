# Sample data for stack testing

This folder holds **sample OTEL traces** and **outputs** from ingestion and evidence gathering when you run the stack locally or in CI.

## Contents

- **`example_otel_2.json`** – Sample Observe SDK OTEL trace (array of spans). Used by:
  - `./ingest_otel.sh` (default file)
  - `./ingest_and_evidence.sh` (default file)

- **`output/`** – Written by the scripts; do not edit.
  - **`ingestion_response.json`** – Response from the ingestion (extraction) API.
  - **`evidence_response.json`** – Response from the evidence gathering API.

## Usage

From the repo root:

```bash
./scripts/ingest_otel.sh                          # uses sample-data/example_otel_2.json, saves to sample-data/output/ingestion_response.json
./scripts/ingest_otel.sh sample-data/other.json   # custom file
./scripts/run_evidence.sh                          # saves to sample-data/output/evidence_response.json
./scripts/ingest_and_evidence.sh                   # ingest default sample then run evidence; both outputs under sample-data/output/
```

## Best practice

- Keep sample traces here (not in repo root) so production configs and one-off data stay separate.
- Add new samples as needed; avoid committing very large files unless necessary for reproducibility.
- Outputs under `output/` are overwritten each run; add `output/` to `.gitignore` if you do not want to commit them.
