# OTel Ingestion Cognitive Agent

A cognitive agent that ingests OpenTelemetry (OTel) trace data and extracts entities, relationships, and knowledge graphs.

## Setup

### Environment Variables

Create a `.env` file in the project root with:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Optional
ENABLE_EMBEDDINGS=true
ENABLE_DEDUP=true
SIMILARITY_THRESHOLD=0.95
```

### Running with Poetry

```bash
# Install dependencies
poetry install

# Run the server
poetry run python ingestion-cognitive-agent/src/telemetry_extraction.py
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

