# IoC CFN Cognitive Agents

A collection of cognitive agents for processing OpenTelemetry data and evidence gathering for reasoning systems.

## Agents

- **[Ingestion Cognitive Agent](ingestion-cognitive-agent/)** (port 8086) - Extracts knowledge from OpenTelemetry traces
- **[Evidence Gathering Agent](evidence-gathering-agent/)** (port 8087) - Retrieves relevant evidence from knowledge graphs

## Quick Start

```bash
# Install dependencies
poetry install

# Run ingestion agent
cd ingestion-cognitive-agent && poetry run uvicorn app.main:app --port 8086

# Run evidence gathering agent
cd evidence-gathering-agent && poetry run uvicorn app.main:app --port 8087
```

See agent-specific READMEs for detailed documentation.

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

```bash
# Pull latest (main branch)
docker pull ghcr.io/<org>/ingestion-cognitive-agent:latest
docker pull ghcr.io/<org>/evidence-gathering-agent:latest

# Pull specific version (production)
docker pull ghcr.io/<org>/ingestion-cognitive-agent:v1.0.0
docker pull ghcr.io/<org>/evidence-gathering-agent:v1.0.0

# Run containers
docker run -p 8086:8086 ghcr.io/<org>/ingestion-cognitive-agent:latest
docker run -p 8087:8087 ghcr.io/<org>/evidence-gathering-agent:latest
```

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

```
┌─────────────────────────────────────────────────────────┐
│                   OpenTelemetry Traces                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  Ingestion Cognitive Agent │
         │       (Port 8086)          │
         │  - Extract entities        │
         │  - Generate embeddings     │
         │  - Build relations         │
         └────────────┬───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Knowledge Graph│
              │   (Neo4j/DB)   │
              └───────┬────────┘
                      │
                      ▼
      ┌────────────────────────────────┐
      │ Evidence Gathering Agent       │
      │       (Port 8087)              │
      │  - Query decomposition         │
      │  - Path finding                │
      │  - Evidence ranking            │
      └────────────────────────────────┘
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
