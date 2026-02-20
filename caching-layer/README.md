# Caching Layer Service

A placeholder microservice that models the future caching layer API. It mirrors the structure of the other agents in this workspace (FastAPI app, dependencies module, agent package, configuration module, and tests) but all operations are intentionally simple so you can evolve it incrementally.

## Features

- FastAPI server with `/` and `/health` endpoints
- Versioned router at `/api/v1/cache` with:
  - `GET /status` to return stubbed cache metadata
  - `POST /prime` to simulate cache priming requests
  - `POST /store` to persist text or caller-provided vectors in FAISS
  - `POST /search` to perform nearest-neighbor lookups over cached entries
- Lightweight service class (`CacheOrchestratorService`) that owns in-memory stats and health reporting hooks
- Unit and integration tests to keep the skeleton wired into CI from day one
- Taskfile for common dev loops (run server, run tests, lint)

## Project Layout

```
caching-layer/
├── Taskfile.yml
├── app/
│   ├── main.py
│   ├── dependencies.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── agent/
│   │   └── service.py
│   ├── config/
│   │   └── settings.py
│   └── data/
│       └── base.py
└── tests/
    ├── conftest.py
    ├── unit/
    │   └── test_agent.py
    └── integration/
        └── test_api.py
```

## Running Locally

```bash
# Install project deps from repository root
poetry install

# Run the caching layer service
cd caching-layer
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8091 --reload
```

The service exposes docs at `http://localhost:8091/docs`.

## Taskfile Shortcuts

```bash
# View all tasks
cd caching-layer && task

# Run tests
cd caching-layer && task test

# Start the API with hot reload
cd caching-layer && task run:dev
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `SERVICE_NAME` | `caching-layer-service` | Service identifier shown in logs |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8091` | Bind port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `CACHE_NAMESPACE` | `default` | Logical namespace for cache keys |
| `DEFAULT_CACHE_TTL_SECONDS` | `300` | Default TTL applied to simulated cache entries |
| `CACHE_VECTOR_DIMENSION` | `1536` | Embedding dimension for the FAISS index |
| `CACHE_METRIC` | `l2` | FAISS distance metric (`l2` or `ip`) |

## Demo Script

Run the one-shot demo that boots the FastAPI service, stores three knowledge
snippets, and performs a similarity search against them:

```bash
cd caching-layer
chmod +x scripts/demo_store_search.sh
./scripts/demo_store_search.sh
```

The script waits for the service to become healthy, uses the `/api/v1/cache/store`
endpoint for each snippet, then queries `/api/v1/cache/search` with `"vector search"`
to show the nearest matches.

## Next Steps

- Swap the stub service for a real cache backplane or SDK
- Replace the in-memory stats collector with metrics exporters
- Flesh out the schemas and persistence layer to integrate with upstream agents
