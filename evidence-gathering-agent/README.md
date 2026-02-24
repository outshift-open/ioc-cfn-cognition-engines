# Evidence Gathering Agent

A small, production‑ready FastAPI service that exposes an API for “evidence gathering” logic.  
It follows a layered, testable, DB‑agnostic design so you can run it with a mock data source now and swap in a real database or external service later with no API changes.

## Features
- Clean separation of concerns: API (HTTP) ⟷ Agent logic ⟷ Data repository
- Mock data repository out of the box; real DB client can be added later
- Poetry‑managed project (lockfile included)
- Unit and integration tests
- Dockerfile for containerized deployment

## Project Layout

```
evidence-gathering-agent/
├─ app/
│  ├─ main.py                  # FastAPI app & /health
│  ├─ api/                     # HTTP layer only
│  │  ├─ routes.py             # /api/knowledge-mgmt/reasoning/evidence + graph placeholders
│  │  └─ schemas.py            # Pydantic request/response models
│  ├─ agent/                   # Core agent logic (orchestrator, processors)
│  │  ├─ evidence.py           # Orchestrator: process_evidence(...)
│  │  ├─ single_entity.py      # Single-entity engine
│  │  ├─ multi_entities.py     # Multi-entity engine
│  │  ├─ utiles.py             # GraphSession, PathFormatter, helpers
│  │  ├─ llm_clients.py        # Judge/Ranker/Extractor (Azure + fallbacks)
│  │  ├─ embeddings.py         # Embedding manager (+ embeddings_config.yml)
│  ├─ data/                    # Data access abstraction
│  │  ├─ base.py               # DataRepository Protocol
│  │  └─ mock_repo.py          # Mock implementation (default)
│  ├─ config/
│  │  └─ settings.py           # Env config
│  └─ dependencies.py          # DI: choose which repo to use
├─ tests/
│  ├─ unit/test_agent.py       # unit test for service
│  └─ integration/test_api.py  # API smoke test
├─ samples/
│  └─ post_evidence.sh         # curl example for /reasoning/evidence
├─ scripts/run_local.sh        # convenience launcher
├─ pyproject.toml              # Poetry project config
├─ poetry.lock                 # locked dependencies
├─ .env.example                # sample env
├─ .gitignore
└─ Dockerfile
```

## Requirements
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) (`curl -sSL https://install.python-poetry.org | python3 -`)

## Quick Start (Poetry)
```bash
cd /Users/zahraamini/Documents/TKF_proj/TKF_V2/ioc-cfn-cognitive-agents/evidence-gathering-agent

# optional: start with default env
cp .env.example .env

# install deps
poetry install

# run the service
poetry run uvicorn app.main:app --reload --port 8087
```

Check it’s up:
```bash
curl -s http://localhost:8087/health | jq .
```

Call the API:
```bash
curl -s -X POST http://localhost:8087/api/knowledge-mgmt/reasoning/evidence \
  -H 'Content-Type: application/json' \
  -d '{
    "reasoner_cognition_request_id": "demo-1",
    "intent": "what does the orchestrator do with email agent?",
    "records": [],
    "meta": {}
  }' | jq .
```

Or use the sample script:
```bash
bash evidence-gathering-agent/samples/post_evidence.sh
```

With the default mock repository, the algorithm runs but returns empty evidence (no data to process).

## Evidence pipeline (migrated)

- Orchestrator: `app/agent/evidence.py` (`process_evidence`) mirrors the manager service:
  - Entity extraction (LLM or fallback) → query decomposition (`meta.request_decomposition`)
  - Per subquery: single-entity (`single_entity.py`) or multi-entity (`multi_entities.py`)
  - Accumulates `meta.subquery_results` with symbolic paths and statuses
- Data access is abstracted via `DataRepository`; default `mock_repo.py` returns empty-but-correct shapes
- LLM and embeddings:
  - `llm_clients.py`: Azure-backed clients with deterministic fallbacks if env is missing
  - `embeddings.py`: Hugging Face by default; configurable in `embeddings_config.yml`

## DB-facing placeholder endpoints
These are wired to the repository abstraction and work with the mock repo now. Swap the repo implementation later to point at your real data service.

- POST `/api/v1/graph/paths`
  - Request:
    ```json
    {"source_id":"A","target_id":"B","max_depth":3,"limit":10}
    ```
  - Example:
    ```bash
    curl -s -X POST http://localhost:8087/api/v1/graph/paths \
      -H 'Content-Type: application/json' \
      -d '{"source_id":"A","target_id":"B"}' | jq .
    ```

- GET `/api/v1/graph/neighbors/{concept_id}`
  - Example:
    ```bash
    curl -s http://localhost:8087/api/v1/graph/neighbors/A | jq .
    ```

- POST `/api/v1/graph/concepts/by_ids`
  - Request:
    ```json
    {"ids":["A","B","C"]}
    ```
  - Example:
    ```bash
    curl -s -X POST http://localhost:8087/api/v1/graph/concepts/by_ids \
      -H 'Content-Type: application/json' \
      -d '{"ids":["A","B"]}' | jq .
    ```

## Configuration
Set environment variables in `.env` (or export them) if needed:

- `DATA_LAYER_BASE_URL` – optional base URL of a real data service (when you implement `db_repo.py`)
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` – when you wire LLM clients
- `EG_MAX_DEPTH`, `EG_PATH_LIMIT` – tuning knobs for path search (if used)

## Switching Data Sources
`app/data/base.py` defines a `DataRepository` Protocol:
```python
class DataRepository(Protocol):
    async def neighbors(self, concept_id: str) -> dict: ...
    async def find_paths(self, source_id: str, target_id: str, max_depth: int, limit: int, relations: list[str] | None = None) -> dict: ...
    async def get_concepts_by_ids(self, ids: list[str]) -> list[dict]: ...
```
The default `MockDataRepository` satisfies this interface. To use a real DB, implement `DbRepository` with these methods (e.g., calling your TKF/data‑logic API) and switch DI in `app/dependencies.py`:
```python
from .data.mock_repo import MockDataRepository
# from .data.db_repo import DbRepository

def get_repository():
    return MockDataRepository()  # swap to DbRepository(...)
```

## Tests
```bash
poetry run pytest -q
```

## Docker
```bash
cd evidence-gathering-agent
docker build -t evidence-agent .
docker run --rm -p 8087:8087 --env-file .env evidence-agent
```

