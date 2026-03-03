# IoC CFN Cognitive Agents

A collection of cognitive agents for processing OpenTelemetry data and evidence gathering for reasoning systems.

## Agents

- **[Ingestion Cognitive Agent](ingestion-cognitive-agent/)** – Extracts knowledge from OpenTelemetry traces (entities, relations, embeddings).
- **[Evidence Gathering Agent](evidence-gathering-agent/)** – Retrieves relevant evidence from the knowledge graph (e.g. “What does Miss-Marple do?”).

The evidence agent can use an optional **mocked DB** (Neo4j-backed graph API). For that setup, run the mocked-db service and set `DATA_LAYER_BASE_URL` or `MOCKED_DB_BASE_URL`; see [evidence-gathering-agent/README.md](evidence-gathering-agent/README.md). When running via the **unified gateway** (Docker or local), the in-memory cache is used and no external data layer is required.

## Quick Start

### Run the gateway with Docker (recommended)

The gateway serves both ingestion and evidence on **port 9004**:

```bash
# From repo root
docker compose up --build
```

Then use the API at `http://localhost:9004`:

| Backend   | Path | Example |
|-----------|------|--------|
| Gateway health | `/health` | `GET http://localhost:9004/health` |
| Ingestion | `/api/knowledge-mgmt/extraction` | `POST http://localhost:9004/api/knowledge-mgmt/extraction` |
| Evidence  | `/api/knowledge-mgmt/reasoning/evidence` | `POST http://localhost:9004/api/knowledge-mgmt/reasoning/evidence` |

**Confluence paths** (above); prefixed paths also work: `/ingestion/...`, `/evidence/...`. Cache is in-process only (not exposed); ingestion and evidence share it inside the container.

### Run the gateway locally (no Docker)

One-time setup so the gateway can import ingestion, evidence, and caching:

```bash
./scripts/setup_local_links.sh
```

Then:

```bash
PYTHONPATH=. poetry run uvicorn gateway.app.main:app --host 0.0.0.0 --port 9004
```

Use `http://localhost:9004` as the base URL. See [gateway/README.md](gateway/README.md) for more.

### Run agents individually (development)

To run ingestion or evidence as separate services (e.g. for development), see each agent's README. Typically you use the gateway for a single entrypoint.

---

## CI/CD Workflow

### 🔄 Automated Docker Builds

The CI pipeline automatically builds and publishes Docker images for both agents using GitHub Actions.

#### **Pull Request** (Build Validation)
When you open a PR:
```bash
git checkout -b feature/my-changes
git push origin feature/my-changes
# Open PR on GitHub
```

**What happens:**
- ✅ Runs unit tests
- ✅ Builds Docker images for **both agents** (validation only)
- ❌ Does **NOT** push images to registry
- 🎯 Purpose: Catch Docker build regressions early

#### **Merge to Main** (Latest Release)
When you merge to `main`:
```bash
git checkout main
git pull origin main
git merge feature/my-changes
git push origin main
```

**What happens:**
- ✅ Runs unit tests
- ✅ Builds Docker images for both agents
- ✅ Pushes with `latest` tag to GHCR

**Published images:**
```
ghcr.io/<org>/ingestion-cognitive-agent:latest
ghcr.io/<org>/evidence-gathering-agent:latest
```

#### **Tag Push** (Versioned Release)
To create a production release:
```bash
# Create and push a semantic version tag
git tag v1.0.0
git push origin v1.0.0
```

**What happens:**
- ✅ Validates tag follows semantic versioning (`vX.Y.Z`)
- ✅ Runs unit tests
- ✅ Builds Docker images for both agents
- ✅ Pushes with version tag to GHCR

**Published images:**
```
ghcr.io/<org>/ingestion-cognitive-agent:v1.0.0
ghcr.io/<org>/evidence-gathering-agent:v1.0.0
```

**Valid tag formats:**
- `v1.0.0` - Standard release
- `v2.3.4-alpha.1` - Pre-release
- `v1.0.0-beta` - Beta release
- `v3.2.1-rc.2` - Release candidate

**Invalid tags will fail CI:**
- `1.0.0` (missing `v` prefix)
- `v1.2` (incomplete version)
- `release-1` (not semver)

### 📦 Using Published Images

The recommended way to run is the **unified gateway image** (single process, port 9004):

```bash
# Pull and run the unified gateway (ingestion + evidence on port 9004)
docker pull ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
docker run -p 9004:9004 ghcr.io/<org>/ioc-cfn-cognitive-agents:latest
```

Then use `http://localhost:9004` for ingestion and evidence paths (see Quick Start).

### 🏗️ Multi-Platform Support

All images are built for:
- `linux/amd64` (x86_64)
- `linux/arm64` (Apple Silicon, ARM servers)

### 🚀 Release Checklist

1. **Ensure tests pass**: `poetry run pytest`
2. **Update version in code** (if needed)
3. **Create semantic version tag**: `git tag v1.0.0`
4. **Push tag**: `git push origin v1.0.0`
5. **Monitor CI**: Check GitHub Actions for build status
6. **Verify images**: `docker pull ghcr.io/<org>/ingestion-cognitive-agent:v1.0.0`

---

## Development

### Prerequisites

- Python 3.11+
- Poetry
- Docker (for containerized development)
- Task (optional, for automation)

### Environment Setup

Create a `.env` file with required credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Knowledge Processing
ENABLE_EMBEDDINGS=true
ENABLE_DEDUP=true
SIMILARITY_THRESHOLD=0.95
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific agent tests
cd ingestion-cognitive-agent && poetry run pytest
cd evidence-gathering-agent && poetry run pytest
```

### Code Quality

```bash
# Linting
poetry run ruff check .

# Auto-fix
poetry run ruff check --fix .

# Format
poetry run ruff format .
```

---

## Architecture

The **unified gateway** runs ingestion and evidence in one process with a shared in-memory cache (port 9004):

```
┌─────────────────────────────────────────────────────────┐
│                   OpenTelemetry Traces                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────────────────┐
         │  Unified Gateway (port 9004)                    │
         │  ┌─────────────────────┐  ┌──────────────────┐ │
         │  │ Ingestion            │  │ Evidence         │ │
         │  │ - Extract entities   │  │ - Query intent   │ │
         │  │ - Generate embeddings│  │ - Path finding   │ │
         │  │ - Build relations    │  │ - Evidence rank  │ │
         │  └──────────┬───────────┘  └────────┬─────────┘ │
         │             │    In-memory cache     │           │
         │             └───────────────┬────────┘           │
         └─────────────────────────────┼───────────────────┘
                                       │
                      (optional)       ▼
              ┌───────────────────────────────┐
              │ External graph (e.g. Neo4j)   │
              │ when DATA_LAYER_BASE_URL set  │
              └───────────────────────────────┘
```

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and add tests
3. Ensure tests pass: `poetry run pytest`
4. Push and open a PR (CI will validate build)
5. After merge, `latest` images are auto-published
6. Tag releases with semantic versions for production

## License

[Add your license here]
