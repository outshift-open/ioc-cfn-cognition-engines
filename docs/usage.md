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
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor
from ingestion.app.config.settings import Settings

settings = Settings()

# Initialize services (requires Azure OpenAI credentials)
concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_api_key=settings.azure_openai_api_key,
    azure_api_version=settings.azure_openai_api_version,
    azure_deployment=settings.azure_openai_deployment,
)

# For testing without Azure OpenAI credentials, use mock_mode:
# concept_service = ConceptRelationshipExtractionService(mock_mode=True)

vector_store = ConceptVectorStore()
processor = KnowledgeProcessor(enable_embeddings=False, enable_dedup=False)

# Sample trace data (observe-sdk-otel format)
payload_data = [
    {
        "Timestamp": "2026-03-17 18:30:00.000000000",
        "TraceId": "trace_001",
        "SpanId": "span_001",
        "ParentSpanId": "",
        "SpanName": "user.service",
        "SpanKind": "Server",
        "ServiceName": "user-service",
        "SpanAttributes": {
            "agent_id": "user_agent",
            "http.method": "GET",
            "http.route": "/users",
            "gen_ai.request.model": "gpt-4o"
        },
        "Duration": 1500000
    }
]

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
import numpy as np
from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer

# Initialize repository
repo = MockDataRepository()

# Create cache layer with custom embedding function
def simple_embed(text: str) -> np.ndarray:
    """Simple embedding function for testing/development."""
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    np.random.seed(hash_val % (2**32))
    return np.random.rand(384).astype(np.float32)

cache_layer = CachingLayer(
    vector_dimension=384,
    metric="l2",
    embed_fn=simple_embed
)

# Populate cache with concepts
concepts = [
    {"name": "user-service", "type": "Service", "description": "User management"},
    {"name": "auth-service", "type": "Service", "description": "Authentication"},
]

for concept in concepts:
    cache_layer.store_knowledge(
        text=f"{concept['name']}: {concept['description']}",
        metadata=concept
    )

# Create evidence request
request = ReasonerCognitionRequest(
    header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
    request_id="evidence-001",
    payload=RequestPayload(
        intent="How does the user service authenticate?"
    )
)

# Gather evidence
async def gather_evidence():
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)
    print(f"Found {len(response.records)} evidence records")
    for record in response.records:
        print(f"  Record: {record.content}")
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

## Testing

### Mock Mode for CI/CD

For testing without Azure OpenAI credentials, use `mock_mode=True`:

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService

# Initialize with mock mode (generates deterministic test data)
concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=None,
    azure_api_key=None,
    mock_mode=True,  # No LLM calls - uses mock data generation
)

# Extract using mock mode (returns mock concepts and relationships)
result = concept_service.extract_concepts_and_relationships(
    payload_data,
    request_id="test-001"
)

print(f"Mock extracted {len(result['concepts'])} concepts")
```

### Integration Tests

The repository includes comprehensive integration tests:

**Mock Tests** (CI-safe, no credentials required):
```bash
# Run tests that work without Azure OpenAI credentials
PYTHONPATH=. poetry run pytest tests/integration/test_usage_examples_mock.py -v
```

**Live Tests** (requires Azure OpenAI credentials):
```bash
# Run tests with real LLM calls (requires .env file with credentials)
PYTHONPATH=. poetry run pytest tests/integration/test_usage_examples_live.py -v -m ''
```

See [tests/integration/README.md](../tests/integration/README.md) for more details.

---

## API Reference

### ConceptRelationshipExtractionService

**Class:** `ingestion.app.agent.service.ConceptRelationshipExtractionService`

**Constructor:**
```python
ConceptRelationshipExtractionService(
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_deployment: str = "gpt-4o",
    azure_api_version: str = "2024-08-01-preview",
    mock_mode: bool = False,
)
```

**Methods:**
- `extract_concepts_and_relationships(payload_data, request_id=None, format_descriptor="observe-sdk-otel")` - Extract concepts and relationships from OpenTelemetry data

### CachingLayer

**Class:** `caching.app.agent.caching_layer.CachingLayer`

**Constructor:**
```python
CachingLayer(
    vector_dimension: int = 1536,
    metric: str = "l2",
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
)
```

**Methods:**
- `store_knowledge(text=None, vector=None, metadata=None)` - Store text or vector with optional metadata
- `search_similar(text=None, vector=None, k=5)` - Find top-k similar items
- `describe()` - Get cache statistics

### ReasonerCognitionRequest

**Class:** `evidence.app.api.schemas.ReasonerCognitionRequest`

**Constructor:**
```python
ReasonerCognitionRequest(
    header: Header,
    request_id: str,
    payload: RequestPayload,
)
```

**Classes:**
- `Header(workspace_id: str, mas_id: str, agent_id: str)`
- `RequestPayload(intent: str)`

---

## Support

- GitHub Issues: https://github.com/cisco-eti/ioc-cfn-cognitive-agents/issues
- Documentation: See `README.md` in the repository
