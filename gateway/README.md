# Gateway

Single entrypoint that forwards requests to **ingestion** and **evidence** agents by path prefix. The **cache** is internal only and is not exposed through the gateway.

## How external clients call the gateway

Clients use **one base URL** and **path prefixes**; the gateway forwards to the right backend.

| Backend   | Path prefix | Example (replace `BASE` with your gateway URL) |
|-----------|-------------|-------------------------------------------------|
| Ingestion | `/ingestion` | `GET BASE/ingestion/health`, `POST BASE/ingestion/api/v1/...` |
| Evidence  | `/evidence`  | `GET BASE/evidence/health`, `POST BASE/evidence/api/knowledge-mgmt/reasoning/evidence` |

**Cache** is not exposed. Evidence and ingestion reach the caching service via `CACHE_BASE_URL` (e.g. `http://caching:8091`) on the internal Docker network only.

**Base URL examples:**

- Local: `http://localhost:8000`
- Deployed: `https://your-gateway.example.com` (TLS in production)

**Example (evidence from an external client):**

```bash
BASE_URL="http://localhost:8000"   # or your deployed gateway URL

curl -s -X POST "${BASE_URL}/evidence/api/knowledge-mgmt/reasoning/evidence" \
  -H "Content-Type: application/json" \
  -d '{"header":{"workspace_id":"w1","mas_id":"m1"},"request_id":"r1","payload":{"intent":"..."}}'
```

No other ports or hostnames are needed; all traffic goes to the gateway.

## Environment (Docker Compose)

The cache URL is **internal only**. The gateway does not expose the cache; it is used only by evidence and ingestion:

- **`CACHE_BASE_URL`** – Base URL of the caching layer (default: `http://caching:8091`). Set in docker-compose for **evidence** (as `CACHING_LAYER_BASE_URL`) and **ingestion** (as `CACHE_BASE_URL`). Both services talk to the cache directly on the Docker network; the gateway does not proxy cache traffic.

Set it in a `.env` next to `docker-compose.yml` or when running compose if you need to override the default.

## Run with Docker Compose

From repo root: `docker compose up --build`. The gateway service exposes port 8000.
