# OTel Ingestion Cognitive Agent

A cognitive agent that ingests OpenTelemetry (OTel) trace data and custom raw messages data and extracts entities, relationships, and knowledge graphs.

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
│   │   ├── service.py          # Extraction services (Telemetry + ConceptRelationship)
│   │   ├── knowledge_processor.py  # Embedding generation & dedup
│   │   └── concept_vector_store.py # HTTP client for caching-layer FAISS service
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

# FAISS Vector Store (in-process via caching-layer library)
ENABLE_FAISS_STORAGE=true
FAISS_VECTOR_DIMENSION=384
FAISS_METRIC=l2

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

### Knowledge Extraction

`POST /api/knowledge-mgmt/extraction`

The extraction endpoint. Accepts a structured request envelope with a header, optional request ID, and a payload containing metadata and trace data. All formats are processed through the LLM-based concept and relationship extraction pipeline.

```bash
curl -X POST http://localhost:8086/api/knowledge-mgmt/extraction \
  -H "Content-Type: application/json" \
  -d @request.json
```

### File-based Extraction (Dev/Testing)

```bash
curl "http://localhost:8086/api/v1/extract/entities_and_relations/from_file?file_path=/path/to/otel.json&save_output=true"
curl "http://localhost:8086/api/v1/extract/concepts_and_relationships/from_file?file_path=/path/to/otel.json"
```

### Health & Metrics

```bash
curl http://localhost:8086/health
curl http://localhost:8086/api/v1/metrics
```

## Request Format

The extraction endpoint accepts the following envelope:

```json
{
  "header": {
    "workspace_id": "ws-123",
    "mas_id": "mas-456",
    "agent_id": "agent-789"
  },
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "payload": {
    "metadata": {
      "format": "observe-sdk-otel"
    },
    "data": [
      {
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "2b6a701a27797f5c",
        "ParentSpanId": "",
        "SpanName": "farm_agent.build_graph.agent",
        "SpanKind": "Client",
        "ServiceName": "corto.farm_agent",
        "SpanAttributes": {
          "agent_id": "farm_agent.build_graph",
          "gen_ai.request.model": "gpt-4"
        },
        "Duration": 21346166
      }
    ]
  }
}
```

| Field | Required | Description |
|---|---|---|
| `header.workspace_id` | Yes | Workspace identifier |
| `header.mas_id` | Yes | MAS identifier |
| `header.agent_id` | No | Agent identifier |
| `request_id` | No | Client-supplied UUID; echoed back as `response_id`. Generated if absent. |
| `payload.metadata.format` | Yes | Data format: `observe-sdk-otel` or `openclaw` |
| `payload.metadata.*` | No | Additional metadata fields become Knowledge Graph labels |
| `payload.data` | Yes | Array of trace records matching the declared format |

## Response Format

### Success

```json
{
  "header": {
    "workspace_id": "ws-123",
    "mas_id": "mas-456",
    "agent_id": "agent-789"
  },
  "response_id": "550e8400-e29b-41d4-a716-446655440000",
  "concepts": [
    {
      "id": "15118c8b99e5813a2239279f0d7fb7c6",
      "name": "website_selector_agent",
      "description": "Agent: website_selector_agent",
      "type": "concept",
      "attributes": {
        "concept_type": "agent",
        "embedding": [[0.042, 0.018, -0.079]]
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
      "relationship": "SENDS_PROMPT_TO",
      "attributes": {
        "source_name": "website_selector_agent",
        "target_name": "gpt-4o",
        "summarized_context": "Agent sends inference request to the LLM."
      }
    }
  ],
  "descriptor": "observe-sdk-otel",
  "metadata": {
    "records_processed": 659,
    "concepts_extracted": 10,
    "relations_extracted": 16,
    "dedup_enabled": false,
    "concepts_deduped": 0,
    "relations_deduped": 0
  }
}
```

### Error

When processing fails, the response contains an `error` block instead of concepts/relations:

```json
{
  "header": {
    "workspace_id": "ws-123",
    "mas_id": "mas-456",
    "agent_id": "agent-789"
  },
  "response_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": {
    "message": "An end-user meaningful error description.",
    "detail": {
      "traceback": "..."
    }
  }
}
```

## Architecture

### Components

- **ConceptRelationshipExtractionService** (`agent/service.py`): Extraction service used by `/api/knowledge-mgmt/extraction`. Distils traces into a compact payload and delegates concept/relationship identification to the LLM in a two-stage pipeline (concepts first, then relationships).
- **KnowledgeExtractionService** (`agent/service.py`): Deterministic extraction service that builds a graph from span hierarchy and attributes. Used by the file-based dev endpoints.
- **KnowledgeProcessor** (`agent/knowledge_processor.py`): Handles embedding generation and semantic deduplication of extracted concepts and relations.
- **ConceptVectorStore** (`agent/concept_vector_store.py`): Imports the caching-layer's `CachingLayer` as a library and uses its FAISS index in-process to store extracted concepts with their embeddings.
- **DataRepository** (`data/base.py`): Protocol for data access abstraction.
- **Settings** (`config/settings.py`): Environment-based configuration using Pydantic.

### Pipeline

```
POST request
  → ConceptRelationshipExtractionService  (extract concepts & relationships)
  → KnowledgeProcessor                    (generate embeddings, deduplicate)
  → ConceptVectorStore                    (store in FAISS via caching-layer)
  → Return response
```

> **Note:** The caching-layer source must be present in the workspace (sibling directory). `FAISS_VECTOR_DIMENSION` must match the embedding model output (384 for `all-MiniLM-L6-v2`).

### Extracted Concepts

- **Queries**: The original user question or request that initiated the trace
- **Agents**: From `agent_id` attribute
- **Services**: From `ServiceName` field
- **LLMs**: From `gen_ai.request.model` attribute
- **Tools**: From `tool_calls` attributes
- **Functions**: From `llm.request.functions.{N}.name` attributes
- **Outputs**: The final answer or artifact produced at the end of the trace
- **Domain concepts**: Higher-level ideas identified from system prompts and function descriptions

### Extracted Relations

- **SENDS_PROMPT_TO**: Agent/Service → LLM
- **INVOKES_TOOL**: LLM → Tool
- **EXECUTES_FUNCTION**: Agent/LLM → Function
- **DELEGATES_TASK_TO**: Parent Agent → Child Agent (from span hierarchy)
- **SUBMITTED_TO**: Query → Agent/Service
- **PRODUCES / ANSWERS**: Agent → Output → Query
- Additional descriptive labels generated by the LLM when available

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

1. Extend `ConceptRelationshipExtractionService` or `TelemetryExtractionService` in `agent/service.py`
2. Add new Pydantic models in `api/schemas.py` if needed
3. Add new endpoints or extend the unified handler in `api/routes.py`

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
