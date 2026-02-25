# TKF Data Layer

Neo4j-backed graph API for the **evidence-gathering-agent**. Exposes paths, neighbors, concepts-by-ids, and optional semantic (vector) search so the evidence agent can query real data instead of the mock.

## Prerequisites

- Python 3.11+
- **Neo4j** running (e.g. Docker: `docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j`)
- KCR data: `mock_db/KCR_sample1.json` (or set `KCR_JSON_PATH`)

## Setup

```bash
cd tkf-data-layer
cp .env.example .env
# Edit .env with your Neo4j credentials (URI, user, password). Never commit or share .env.
poetry install
```

**Credentials:** Store Neo4j and other secrets only in `.env`. The file `.env` is gitignored and must never be committed or shared. Use `.env.example` as a template (it has no real secrets).

## 1. Start Neo4j

```bash
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

## 2. Load KCR data into Neo4j

Either call the admin API after starting the server (see below), or run the loader script:

```bash
# From tkf-data-layer dir; uses mock_db/KCR_sample1.json by default
poetry run python scripts/load_kcr.py
```

Or with custom path:

```bash
KCR_JSON_PATH=/path/to/KCR_sample1.json poetry run python scripts/load_kcr.py
```

## 3. Start the TKF Data Layer server

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8088
```

## 4. (Optional) Load KCR and generate embeddings via API

If you didn’t run the loader script, you can load KCR via the admin endpoint (POST the JSON body with `concepts` and `relations`). Then generate embeddings:

```bash
# Load KCR (body = full KCR JSON)
curl -X POST http://localhost:8088/api/v1/admin/load-kcr -H "Content-Type: application/json" -d @../mock_db/KCR_sample1.json

# Generate embeddings for all concepts
curl -X POST http://localhost:8088/api/v1/admin/generate-embeddings
```

## 5. Point evidence-gathering-agent at this server

In the evidence-gathering-agent, set:

- `TKF_DATA_LAYER_BASE_URL=http://localhost:8088` (or your tkf-data-layer URL)

and use the HTTP DataRepository (see evidence-gathering-agent README).

## API (aligned with evidence-gathering-agent)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/graph/paths` | Find paths between two concepts |
| GET | `/api/v1/graph/neighbors/{concept_id}` | Neighbors of a concept |
| POST | `/api/v1/graph/concepts/by_ids` | Bulk concept lookup |
| POST | `/api/v1/semantic/similar` | Vector search (query_vector, k) |
| POST | `/api/v1/admin/load-kcr` | Load KCR JSON |
| POST | `/api/v1/admin/clear` | Clear graph |
| POST | `/api/v1/admin/generate-embeddings` | Generate embeddings for all concepts |

Request/response shapes match the evidence-gathering-agent’s `DataRepository` contract.
