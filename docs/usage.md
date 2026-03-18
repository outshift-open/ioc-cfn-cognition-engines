# Cognition Engine - Usage Guide

## Installation

```bash
# From Artifactory
pip install cognition-engine --extra-index-url https://your-artifactory-url/artifactory/api/pypi/outshift-pypi/simple

# From local build
poetry build
pip install dist/cognition_engine-0.1.0-py3-none-any.whl
```

## Embedding Model Setup

The package uses `fastembed` for embeddings. **Model is NOT bundled** - downloads automatically (~100MB on first use).

**Option 1: Auto-download** (default):
```python
from evidence.app.agent.embeddings import EmbeddingManager
manager = EmbeddingManager()  # Auto-downloads BAAI/bge-small-en-v1.5 to ~/.cache/fastembed/
```

**Option 2: Pre-download** (for corporate networks):
```bash
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

---

## Example 1: Knowledge Extraction

Extract concepts and relationships from OpenTelemetry traces.

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.processors import KnowledgeProcessor
from ingestion.app.config.settings import Settings

settings = Settings()

# Initialize services
concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_api_key=settings.azure_openai_api_key,
    azure_api_version=settings.azure_openai_api_version,
    azure_deployment=settings.azure_openai_deployment,
)
vector_store = ConceptVectorStore()
processor = KnowledgeProcessor()

# Sample trace data
payload_data = [{
    "resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "user-service"}}]},
        "scopeSpans": [{
            "spans": [{
                "name": "GET /users",
                "kind": "SPAN_KIND_SERVER",
                "attributes": [
                    {"key": "http.method", "value": {"stringValue": "GET"}},
                    {"key": "http.route", "value": {"stringValue": "/users"}}
                ]
            }]
        }]
    }]
}]

# Extract → Process → Store
result = concept_service.extract_concepts_and_relationships(
    payload_data, request_id="req-001", format_descriptor="observe-sdk-otel"
)
result = processor.process(result)
vector_store.store_concepts(result.get("concepts", []))

print(f"Extracted {len(result['concepts'])} concepts, {len(result['relations'])} relationships")

# Search similar concepts
similar = vector_store.find_similar_concepts("user service", top_k=3)
for concept, score in similar:
    print(f"  {concept['name']} (similarity: {score:.3f})")
```

---

## Example 2: Evidence Gathering

Gather evidence from the knowledge graph using natural language queries.

```python
import asyncio
from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, ReasonerPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer
from caching.app.models.cache_models import ConceptData

# Initialize repository and cache
repo = MockDataRepository()
cache_layer = CachingLayer()

# Populate cache with concepts
for concept_data in [
    ConceptData(name="user-service", concept_type="Service", description="User management"),
    ConceptData(name="auth-service", concept_type="Service", description="Authentication"),
]:
    cache_layer.add_concept(concept_data)

# Create evidence request
request = ReasonerCognitionRequest(
    header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
    request_id="evidence-001",
    payload=ReasonerPayload(
        intent="How does the user service authenticate?",
        metadata={"max_depth": 3, "path_limit": 10}
    )
)

# Gather evidence
async def gather_evidence():
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)
    print(f"Found {len(response.records)} evidence records")
    for record in response.records:
        print(f"  {record.source_concept} --[{record.relation_type}]--> {record.target_concept}")
    return response

response = asyncio.run(gather_evidence())
```

---

## Example 3: Management Plane Registration

Register cognition engines with the IOC management plane (only needed when deploying as a service).

```python
import asyncio
from gateway import register_both_engines

async def register_engines():
    result = await register_both_engines(
        mgmt_plane_url="http://localhost:9000",
        engine_host="cognition-engine.example.com",
        engine_port=9004,
        workspace_name="Default Workspace"
    )

    print(f"Knowledge Management: {result['knowledge_management']['status']}")
    print(f"Semantic Negotiation: {result['semantic_negotiation']['status']}")

asyncio.run(register_engines())
```

**Individual registration** (if you only need one engine):
```python
from gateway import register_knowledge_management_engine, register_semantic_negotiation_engine

# Register Knowledge Management only
result = await register_knowledge_management_engine(
    mgmt_plane_url="http://localhost:9000",
    engine_host="localhost",
    engine_port=9004
)
```

---

## Environment Configuration

Create a `.env` file:

```bash
# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# LLM Configuration
LLM_PROVIDER=azure-openai
LLM_MODEL=azure/gpt-4o

# Optional: External data layer (defaults to MockDataRepository)
DATA_LAYER_BASE_URL=http://localhost:8088

# Optional: Evidence gathering tuning
EG_MAX_DEPTH=4
EG_PATH_LIMIT=20

# Optional: SSL bypass for corporate proxies
HTTPX_VERIFY=false
OPENAI_VERIFY_SSL=false
```

---

## Support

- GitHub Issues: https://github.com/cisco-eti/ioc-cfn-cognitive-agents/issues
- Documentation: See `README.md` in the repository
