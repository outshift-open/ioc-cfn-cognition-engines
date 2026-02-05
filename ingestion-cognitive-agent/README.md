# OTel Ingestion Cognitive Agent

A cognitive agent that ingests OpenTelemetry (OTel) trace data and extracts entities, relationships, and knowledge graphs.

## Project Structure

```
ingestion-cognitive-agent/
├── Taskfile.yml                # Task automation
├── app/
│   ├── main.py                 # FastAPI app entry point & /health endpoint
│   ├── dependencies.py         # Dependency injection configuration
│   │
│   ├── api/                    # HTTP layer only
│   │   ├── routes.py           # API endpoints (/api/v1/...)
│   │   └── schemas.py          # Pydantic request/response models
│   │
│   ├── agent/                  # Core agent logic
│   │   ├── base.py             # AdapterSDK base class
│   │   ├── service.py          # TelemetryExtractionService
│   │   └── knowledge_processor.py
│   │
│   ├── data/                   # Data access abstraction
│   │   ├── base.py             # DataRepository Protocol
│   │   └── mock_repo.py        # File-based mock implementation
│   │
│   └── config/
│       └── settings.py         # Environment settings (Pydantic)
│
└── tests/
    ├── conftest.py             # Shared fixtures
    ├── unit/
    │   └── test_agent.py       # Unit tests for services
    └── integration/
        └── test_api.py         # API smoke tests
```

## Setup

### Environment Variables

Create a `.env` file in the project root with:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Knowledge Processing Options
ENABLE_EMBEDDINGS=true
ENABLE_DEDUP=true
SIMILARITY_THRESHOLD=0.95

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8086
LOG_LEVEL=INFO
```

### Running with Poetry

```bash
# Install dependencies
poetry install

# Run the server
poetry run python -m app.main

# Or use uvicorn directly
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8086 --reload
```

Server starts on `http://localhost:8086`

### Running with Docker

```bash
# Build the image (from project root)
docker build --build-arg GITHUB_TOKEN=$GITHUB_TOKEN -t otel-ingestion-agent .

# Run the container
docker run -p 8086:8086 --env-file .env otel-ingestion-agent

# Or run with inline env vars
docker run -p 8086:8086 \
  -e AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/ \
  -e AZURE_OPENAI_API_KEY=your-api-key \
  -e AZURE_OPENAI_DEPLOYMENT=gpt-4o \
  otel-ingestion-agent
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8086/health
```

### Get Metrics

```bash
curl http://localhost:8086/api/v1/metrics
```

### Extract from File

```bash
curl "http://localhost:8086/api/v1/extract/entities_and_relations/from_file?file_path=/path/to/otel.json&save_output=true"
```

### Extract from Request Body (Batch)

```bash
curl -X POST http://localhost:8086/api/v1/extract/entities_and_relations/batch \
  -H "Content-Type: application/json" \
  -d @otel_traces.json
```

## Request Format

Input OTel trace data (JSON array):

```json
[
  {
    "TraceId": "162b29522a339e6b1acb21b8041dcda5",
    "SpanId": "2b6a701a27797f5c",
    "ParentSpanId": "",
    "SpanName": "farm_agent.build_graph.agent",
    "ServiceName": "corto.farm_agent",
    "SpanAttributes": {
      "agent_id": "farm_agent.build_graph",
      "gen_ai.request.model": "gpt-4"
    },
    "Duration": 21346166
  }
]
```

## Response Format

```json
{
  "knowledge_cognition_request_id": "30cbc343f7a41aa2c91bdffc08b99f1f",
  "concepts": [
    {
      "id": "15118c8b99e5813a2239279f0d7fb7c6",
      "name": "website_selector_agent",
      "description": "Agent: website_selector_agent",
      "type": "concept",
      "attributes": {
        "concept_type": "agent",
        "embedding": [[0.042, 0.018, -0.079, ...]]
      }
    },
    {
      "id": "4e706aec50174e58f15a52a53e6ca4f5",
      "name": "gpt-4o",
      "description": "Llm: gpt-4o",
      "type": "concept",
      "attributes": {
        "concept_type": "llm",
        "embedding": [[-0.017, 0.015, 0.014, ...]]
      }
    }
  ],
  "relations": [
    {
      "id": "91b581b91afb041bdcca33d74ab687c2",
      "node_ids": [
        "15118c8b99e5813a2239279f0d7fb7c6",
        "4e706aec50174e58f15a52a53e6ca4f5"
      ],
      "relationship": "USES",
      "attributes": {
        "source_name": "website_selector_agent",
        "target_name": "gpt-4o",
        "summarized_context": "USES interaction"
      }
    },
    {
      "id": "f14216bda104e771287f7b90a21a198f",
      "node_ids": [
        "b9289c4679466c4e15ee79b2dfa55f5a",
        "15118c8b99e5813a2239279f0d7fb7c6"
      ],
      "relationship": "COORDINATES",
      "attributes": {
        "source_name": "Miss-Marple",
        "target_name": "website_selector_agent",
        "summarized_context": "COORDINATES interaction"
      }
    }
  ],
  "descriptor": "telemetry knowledge extraction",
  "meta": {
    "records_processed": 659,
    "concepts_extracted": 10,
    "relations_extracted": 16,
    "dedup_enabled": false,
    "concepts_deduped": 0,
    "relations_deduped": 0
  }
}
```

## Architecture

### Components

- **TelemetryExtractionService** (`agent/service.py`): Core logic for extracting entities and relationships from OTel traces
- **KnowledgeProcessor** (`agent/knowledge_processor.py`): Handles embedding generation and semantic deduplication
- **DataRepository** (`data/base.py`): Protocol for data access abstraction
- **Settings** (`config/settings.py`): Environment-based configuration using Pydantic

### Extracted Entities

- **Agents**: From `agent_id` attribute
- **Services**: From `ServiceName` field
- **LLMs**: From `gen_ai.request.model` attribute
- **Tools**: From `tool_calls` attributes

### Extracted Relations

- **USES**: Agent → LLM
- **CALLS**: LLM → Tool
- **COORDINATES**: Parent Agent → Child Agent (from span hierarchy)

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and sample data
├── unit/
│   └── test_agent.py        # Unit tests for extraction service
└── integration/
    └── test_api.py          # API smoke tests
```

### Running Tests

```bash
# Run all tests
task test

# Or with poetry directly
poetry run pytest tests/ -v

# Run unit tests only
task test:unit

# Run integration tests only
task test:integration

# Run tests with coverage
task test:cov
```

### Test Coverage

Tests cover:
- **Entity extraction**: Agents, services, LLMs, tools
- **Relation extraction**: USES, CALLS, COORDINATES
- **Deduplication**: Name-based and semantic deduplication
- **Embedding generation**: Vector generation and similarity
- **API endpoints**: Health, metrics, batch extraction

## Task Runner

This project uses [Task](https://taskfile.dev/) for automation. Install it with:

```bash
# macOS
brew install go-task

# or with npm
npm install -g @go-task/cli
```

### Available Tasks

```bash
# Show all available tasks
task

# Development
task install          # Install dependencies
task run              # Run server
task run:dev          # Run with hot reload

# Testing
task test             # Run all tests
task test:unit        # Run unit tests
task test:integration # Run integration tests
task test:cov         # Run with coverage report

# Code Quality
task lint             # Run linting
task lint:fix         # Auto-fix lint issues
task format           # Format code
task check            # Run all checks

# Docker
task docker:build     # Build Docker image
task docker:run       # Run container
task docker:stop      # Stop container
task docker:logs      # View logs

# Cleanup
task clean            # Clean temp files
task clean:all        # Clean everything including Docker
```

## Development

### Adding New Extractors

1. Extend `TelemetryExtractionService` in `agent/service.py`
2. Add new Pydantic models in `api/schemas.py` if needed
3. Create new endpoints in `api/routes.py`

### Customizing Data Sources

1. Implement the `DataRepository` protocol from `data/base.py`
2. Update `dependencies.py` to use your implementation

### Running Locally

```bash
# Quick start with Task
task install
task run:dev

# Or manually
poetry install
poetry run uvicorn app.main:app --reload
```
