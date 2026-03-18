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

## Example 1: Knowledge Extraction (Production Setup)

Extract concepts and relationships from OpenTelemetry traces with proper embeddings and caching.

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from caching.app.agent.caching_layer import CachingLayer
from ingestion.app.config.settings import Settings
import os

settings = Settings()

# 1. Initialize embedding manager (production best practice)
# Uses fastembed with BAAI/bge-small-en-v1.5 model
model_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip() or None
embedding_manager = EmbeddingManager(model_path=model_path)

def embed_fn(text: str):
    """Proper embedding function using fastembed."""
    out = embedding_manager.generate_embedding(text)
    if out is None:
        raise ValueError("Embedding returned None")
    return out

# 2. Create shared cache layer with proper embeddings
cache_layer = CachingLayer(
    vector_dimension=384,  # bge-small-en-v1.5 dimension
    metric="l2",
    embed_fn=embed_fn,
)

# 3. Initialize extraction service (requires Azure OpenAI credentials)
concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_api_key=settings.azure_openai_api_key,
    azure_api_version=settings.azure_openai_api_version,
    azure_deployment=settings.azure_openai_deployment,
)

# For testing without Azure OpenAI credentials, use mock_mode:
# concept_service = ConceptRelationshipExtractionService(mock_mode=True)

# 4. Create vector store with shared cache layer
vector_store = ConceptVectorStore(cache_layer=cache_layer)

# 5. Enable embeddings in processor (generates embeddings for concepts)
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)

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

## Example 2: Evidence Gathering (Production Setup)

Gather evidence from the knowledge graph using natural language queries.

```python
import asyncio
import os
import numpy as np
from evidence.app.agent.evidence import process_evidence
from evidence.app.agent.embeddings import EmbeddingManager
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer

# Initialize repository
repo = MockDataRepository()

# PRODUCTION: Use proper embedding manager with fastembed
model_path = os.getenv("EMBEDDING_MODEL_PATH", "").strip() or None
embedding_manager = EmbeddingManager()

def embed_fn(text: str) -> np.ndarray:
    """Production embedding function using fastembed (BAAI/bge-small-en-v1.5)."""
    embedding = embedding_manager.get_embedding(text)
    if embedding is None:
        raise ValueError("Embedding returned None")
    return embedding

cache_layer = CachingLayer(
    vector_dimension=384,  # bge-small-en-v1.5 dimension
    metric="l2",
    embed_fn=embed_fn
)

# TESTING ONLY: Use simple hash-based embeddings (no model download)
# def simple_embed(text: str) -> np.ndarray:
#     """Simple hash-based embedding for testing/development only."""
#     import hashlib
#     hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
#     np.random.seed(hash_val % (2**32))
#     return np.random.rand(384).astype(np.float32)
# cache_layer = CachingLayer(vector_dimension=384, metric="l2", embed_fn=simple_embed)

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

## Best Practices

### Embeddings in Production

**✅ DO: Use `EmbeddingManager` with fastembed**
```python
from ingestion.app.agent.knowledge_processor import EmbeddingManager
from caching.app.agent.caching_layer import CachingLayer

# Production approach - uses BAAI/bge-small-en-v1.5 via fastembed
embedding_manager = EmbeddingManager()
cache_layer = CachingLayer(
    vector_dimension=384,
    embed_fn=lambda text: embedding_manager.generate_embedding(text)
)
```

**Why:**
- fastembed is ONNX-based (no PyTorch/CUDA dependency)
- BAAI/bge-small-en-v1.5 is bundled in Docker images
- Consistent embeddings across ingestion and evidence services
- Production-quality semantic search

**❌ DON'T: Use custom hash-based embeddings in production**
```python
# Testing only - not for production!
def simple_embed(text: str):
    import hashlib
    return hash_based_vector(text)  # Not semantically meaningful
```

**Why not:**
- Hash-based embeddings have no semantic meaning
- Search results will be random/meaningless
- Only useful for testing without model dependencies

### Shared Cache Layer

**✅ DO: Share `CachingLayer` between services**
```python
# Gateway creates one shared cache
cache_layer = create_shared_caching_layer()

# Pass to both services
vector_store = ConceptVectorStore(cache_layer=cache_layer)
await process_evidence(request, cache_layer=cache_layer)
```

**Why:**
- Concepts extracted by ingestion are available to evidence service
- No HTTP overhead (in-memory)
- Unified vector space for all concepts
- Memory efficient (one FAISS index)

**⚠️ IMPORTANT: Cache is NOT automatically shared**

When using these libraries directly in your Python application, you must explicitly pass the same `CachingLayer` instance to both services. If you create two separate `CachingLayer()` instances, they will have separate FAISS indexes and won't share data.

```python
# ❌ WRONG: Creates two separate caches
cache1 = CachingLayer(vector_dimension=384, embed_fn=embed_fn)
cache2 = CachingLayer(vector_dimension=384, embed_fn=embed_fn)
vector_store = ConceptVectorStore(cache_layer=cache1)  # Cache 1
await process_evidence(request, cache_layer=cache2)    # Cache 2 (isolated!)

# ✅ CORRECT: Create once, pass to both
cache_layer = CachingLayer(vector_dimension=384, embed_fn=embed_fn)
vector_store = ConceptVectorStore(cache_layer=cache_layer)
await process_evidence(request, cache_layer=cache_layer)
```

**Note:** The gateway service handles this automatically in its lifespan - it creates one shared cache and attaches it to both sub-apps. But in your own code, you must manage the sharing explicitly.

### Enable Embeddings in Processor

**✅ DO: Enable embeddings when storing concepts**
```python
# Enable embeddings so concepts can be searched
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)
```

**Why:**
- Concepts without embeddings cannot be searched
- ConceptVectorStore.store_concepts() skips concepts without embeddings
- Evidence service needs embeddings for similarity search

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
