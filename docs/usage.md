# Cognition Engine - Usage Guide

## Quick Start

### Installation

```bash
# From Artifactory
pip install cognition-engine --extra-index-url https://your-artifactory-url/artifactory/api/pypi/outshift-pypi/simple

# From local build
poetry build
pip install dist/cognition_engine-0.1.0-py3-none-any.whl
```

### Embedding Model Setup

The package uses `fastembed` for embeddings. **Model is NOT bundled** - downloads automatically (~100MB on first use).

```bash
# Auto-downloads on first use (default)
python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

---

## Common Patterns

### Pattern 1: FastAPI Application with Multiple Endpoints

**Use this pattern when** you have a FastAPI app with separate endpoints for knowledge extraction and evidence gathering.

```python
from fastapi import FastAPI, Depends, Request
from contextlib import asynccontextmanager
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from evidence.app.agent.evidence import process_evidence
from caching.app.agent.caching_layer import CachingLayer
from ingestion.app.config.settings import Settings
import os

# Create shared cache at app startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize embedding manager (SHARED by all endpoints)
    embedding_manager = EmbeddingManager()

    def embed_fn(text: str):
        out = embedding_manager.generate_embedding(text)
        if out is None:
            raise ValueError("Embedding returned None")
        return out

    # 2. Create ONE cache layer (shared across all requests)
    cache_layer = CachingLayer(
        vector_dimension=384,  # bge-small-en-v1.5 dimension
        metric="l2",
        embed_fn=embed_fn,
    )

    # Store in app state (persists across all requests)
    app.state.cache_layer = cache_layer
    app.state.settings = Settings()

    yield

app = FastAPI(lifespan=lifespan)

# Dependency to inject shared cache
def get_cache_layer(request: Request):
    return request.app.state.cache_layer

def get_settings(request: Request):
    return request.app.state.settings

# Endpoint 1: Knowledge Extraction
@app.post("/api/extraction")
async def extract_knowledge(
    data: dict,
    cache_layer: CachingLayer = Depends(get_cache_layer),
    settings: Settings = Depends(get_settings),
):
    """Extract concepts from telemetry data and store in shared cache."""
    concept_service = ConceptRelationshipExtractionService(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_api_key=settings.azure_openai_api_key,
        azure_api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment,
    )
    vector_store = ConceptVectorStore(cache_layer=cache_layer)  # Uses shared cache
    processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)

    # Extract → Process → Store
    result = concept_service.extract_concepts_and_relationships(
        data.get("payload_data"), request_id=data.get("request_id")
    )
    result = processor.process(result)
    vector_store.store_concepts(result.get("concepts", []))

    return {
        "concepts": len(result["concepts"]),
        "relations": len(result["relations"]),
        "cache_size": cache_layer.describe()["ntotal"]
    }

# Endpoint 2: Evidence Gathering
@app.post("/api/evidence")
async def gather_evidence(
    query: dict,
    cache_layer: CachingLayer = Depends(get_cache_layer),
):
    """Gather evidence from the shared cache."""
    from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
    from evidence.app.data.mock_repo import MockDataRepository

    request = ReasonerCognitionRequest(
        header=Header(
            workspace_id=query.get("workspace_id", "default"),
            mas_id=query.get("mas_id", "default"),
            agent_id=query.get("agent_id", "default")
        ),
        request_id=query.get("request_id", "req-001"),
        payload=RequestPayload(intent=query["intent"])
    )

    repo = MockDataRepository()
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)

    return {
        "records": len(response.records),
        "cache_size": cache_layer.describe()["ntotal"]
    }
```

**Key points:**
- ✅ **One `CachingLayer`** created at startup (in `lifespan`)
- ✅ **Stored in `app.state`** (persists across all requests)
- ✅ **Injected via `Depends()`** into both endpoints
- ✅ **Both endpoints share the same cache** - concepts extracted by `/api/extraction` are immediately available to `/api/evidence`

**❌ WRONG: Creating cache in each endpoint**
```python
@app.post("/api/extraction")
async def extract_knowledge(data: dict):
    # ❌ Creates NEW cache every request - data is lost!
    cache_layer = CachingLayer(vector_dimension=384, ...)
    vector_store = ConceptVectorStore(cache_layer=cache_layer)
    # This cache is discarded after the request
```

---

### Pattern 2: Standalone Script (Both Ingestion and Evidence)

**Use this pattern when** you're writing a standalone Python script that needs both ingestion and evidence gathering.

```python
import asyncio
import os
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from evidence.app.agent.evidence import process_evidence
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer
from ingestion.app.config.settings import Settings

settings = Settings()

# 1. Initialize embedding manager (SHARED)
embedding_manager = EmbeddingManager()

def embed_fn(text: str):
    out = embedding_manager.generate_embedding(text)
    if out is None:
        raise ValueError("Embedding returned None")
    return out

# 2. Create ONE shared cache layer
cache_layer = CachingLayer(
    vector_dimension=384,
    metric="l2",
    embed_fn=embed_fn,
)

# 3. Extract and store concepts
concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_api_key=settings.azure_openai_api_key,
    azure_api_version=settings.azure_openai_api_version,
    azure_deployment=settings.azure_openai_deployment,
)
vector_store = ConceptVectorStore(cache_layer=cache_layer)  # Uses shared cache
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)

payload_data = [
    {
        "Timestamp": "2026-03-18 10:00:00.000000000",
        "TraceId": "trace_001",
        "SpanId": "span_001",
        "ParentSpanId": "",
        "SpanName": "authentication.service",
        "SpanKind": "Server",
        "ServiceName": "auth-service",
        "SpanAttributes": {
            "agent_id": "auth_agent",
            "http.method": "POST",
            "http.route": "/login",
            "gen_ai.request.model": "gpt-4o"
        },
        "Duration": 2000000
    }
]

result = concept_service.extract_concepts_and_relationships(
    payload_data, request_id="req-001", format_descriptor="observe-sdk-otel"
)
result = processor.process(result)
vector_store.store_concepts(result.get("concepts", []))

print(f"✓ Extracted {len(result['concepts'])} concepts")
print(f"✓ Cache contains: {cache_layer.describe()['ntotal']} items")

# 4. Gather evidence using the SAME cache
async def gather_evidence():
    repo = MockDataRepository()
    request = ReasonerCognitionRequest(
        header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
        request_id="evidence-001",
        payload=RequestPayload(intent="How does authentication work?")
    )
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)
    print(f"✓ Found {len(response.records)} evidence records")
    return response

asyncio.run(gather_evidence())
```

---

### Pattern 3: Knowledge Extraction Only

**Use this pattern when** you only need to extract and store concepts.

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService
from ingestion.app.agent.concept_vector_store import ConceptVectorStore
from ingestion.app.agent.knowledge_processor import KnowledgeProcessor, EmbeddingManager
from caching.app.agent.caching_layer import CachingLayer
from ingestion.app.config.settings import Settings

settings = Settings()

# Initialize components
embedding_manager = EmbeddingManager()
cache_layer = CachingLayer(
    vector_dimension=384,
    embed_fn=lambda text: embedding_manager.generate_embedding(text),
)

concept_service = ConceptRelationshipExtractionService(
    azure_endpoint=settings.azure_openai_endpoint,
    azure_api_key=settings.azure_openai_api_key,
    azure_api_version=settings.azure_openai_api_version,
    azure_deployment=settings.azure_openai_deployment,
)

vector_store = ConceptVectorStore(cache_layer=cache_layer)
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)

# Extract and store
payload_data = [{"Timestamp": "...", "TraceId": "...", ...}]
result = concept_service.extract_concepts_and_relationships(payload_data)
result = processor.process(result)
vector_store.store_concepts(result.get("concepts", []))

# Search
similar = vector_store.search_similar(text="authentication", k=3)
for item in similar:
    print(f"{item.get('name', item.get('text'))} (score: {item['score']:.3f})")
```

---

### Pattern 4: Evidence Gathering Only

**Use this pattern when** you only need to retrieve evidence from an existing cache.

```python
import asyncio
from evidence.app.agent.evidence import process_evidence
from evidence.app.agent.embeddings import EmbeddingManager
from evidence.app.api.schemas import ReasonerCognitionRequest, Header, RequestPayload
from evidence.app.data.mock_repo import MockDataRepository
from caching.app.agent.caching_layer import CachingLayer

# Initialize cache with existing data
embedding_manager = EmbeddingManager()
cache_layer = CachingLayer(
    vector_dimension=384,
    embed_fn=lambda text: embedding_manager.get_embedding(text)
)

# Populate cache (normally this would be done by ingestion)
cache_layer.store_knowledge(
    text="auth-service: Authentication service",
    metadata={"name": "auth-service", "type": "Service"}
)

# Gather evidence
async def gather_evidence():
    repo = MockDataRepository()
    request = ReasonerCognitionRequest(
        header=Header(workspace_id="ws-1", mas_id="mas-1", agent_id="agent-1"),
        request_id="evidence-001",
        payload=RequestPayload(intent="How does authentication work?")
    )
    response = await process_evidence(request, repo_adapter=repo, cache_layer=cache_layer)
    print(f"Found {len(response.records)} records")
    return response

asyncio.run(gather_evidence())
```

---

## Best Practices

### ✅ Always Share the Cache

When using both ingestion and evidence gathering:

**✅ CORRECT: Create once, share everywhere**
```python
# Create ONE cache
cache_layer = CachingLayer(vector_dimension=384, embed_fn=embed_fn)

# Pass to both services
vector_store = ConceptVectorStore(cache_layer=cache_layer)
await process_evidence(request, cache_layer=cache_layer)
```

**❌ WRONG: Creating multiple caches**
```python
# Creates separate caches - data won't be shared!
cache1 = CachingLayer(...)
cache2 = CachingLayer(...)
vector_store = ConceptVectorStore(cache_layer=cache1)
await process_evidence(request, cache_layer=cache2)
```

### ✅ Use Production Embeddings

**✅ CORRECT: Use EmbeddingManager with fastembed**
```python
from ingestion.app.agent.knowledge_processor import EmbeddingManager

embedding_manager = EmbeddingManager()  # Uses BAAI/bge-small-en-v1.5
cache_layer = CachingLayer(
    vector_dimension=384,
    embed_fn=lambda text: embedding_manager.generate_embedding(text)
)
```

**Why:**
- ONNX-based (no PyTorch/CUDA dependency)
- Production-quality semantic search
- Consistent across services

**❌ WRONG: Hash-based embeddings (testing only)**
```python
def simple_embed(text: str):
    import hashlib
    # Hash-based - no semantic meaning!
    return hash_based_vector(text)
```

### ✅ Enable Embeddings in Processor

```python
# Enable embeddings so concepts can be searched
processor = KnowledgeProcessor(enable_embeddings=True, enable_dedup=False)
```

Without embeddings, concepts cannot be searched via vector similarity.

---

## Testing

### Mock Mode (No Credentials Required)

For CI/CD or testing without Azure OpenAI:

```python
from ingestion.app.agent.service import ConceptRelationshipExtractionService

# Use mock mode - generates deterministic test data
concept_service = ConceptRelationshipExtractionService(mock_mode=True)

result = concept_service.extract_concepts_and_relationships(payload_data)
print(f"Mock extracted {len(result['concepts'])} concepts")
```

### Integration Tests

```bash
# Mock tests (CI-safe, no credentials)
PYTHONPATH=. poetry run pytest tests/integration/test_usage_examples_mock.py -v

# Live tests (requires Azure OpenAI credentials)
PYTHONPATH=. poetry run pytest tests/integration/test_usage_examples_live.py -v -m ''
```

See [tests/integration/README.md](../tests/integration/README.md) for details.

---

## Environment Configuration

Create a `.env` file:

```bash
# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Optional
DATA_LAYER_BASE_URL=http://localhost:8088
EG_MAX_DEPTH=4
EG_PATH_LIMIT=20
```

---

## API Reference

### ConceptRelationshipExtractionService

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

```python
CachingLayer(
    vector_dimension: int = 1536,
    metric: str = "l2",
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
)
```

**Methods:**
- `store_knowledge(text=None, vector=None, metadata=None)` - Store text or vector
- `search_similar(text=None, vector=None, k=5)` - Find top-k similar items
- `describe()` - Get cache statistics (returns `{"dimension": int, "metric": str, "ntotal": int}`)

### ReasonerCognitionRequest

```python
ReasonerCognitionRequest(
    header: Header,
    request_id: str,
    payload: RequestPayload,
)

Header(workspace_id: str, mas_id: str, agent_id: str)
RequestPayload(intent: str)
```

---

## Support

- GitHub Issues: https://github.com/cisco-eti/ioc-cfn-cognitive-agents/issues
- Documentation: See `README.md` in the repository
