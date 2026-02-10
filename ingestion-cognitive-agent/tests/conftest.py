"""
Shared pytest fixtures and test utilities.
"""
import pytest
from typing import List, Dict, Any

from app.agent.service import TelemetryExtractionService
from app.agent.knowledge_processor import KnowledgeProcessor
from app.data.mock_repo import MockDataRepository


# Sample OTel trace data for testing
SAMPLE_OTEL_RECORDS: List[Dict[str, Any]] = [
    {
        "Timestamp": "2025-12-22 18:37:22.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_001",
        "ParentSpanId": "",
        "SpanName": "orchestrator.agent",
        "SpanKind": "Internal",
        "ServiceName": "corto.orchestrator",
        "SpanAttributes": {
            "agent_id": "orchestrator_agent",
            "execution.success": "true",
            "gen_ai.request.model": "gpt-4o"
        },
        "Duration": 21346166
    },
    {
        "Timestamp": "2025-12-22 18:37:23.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_002",
        "ParentSpanId": "span_001",
        "SpanName": "worker.agent",
        "SpanKind": "Internal",
        "ServiceName": "corto.worker",
        "SpanAttributes": {
            "agent_id": "worker_agent",
            "execution.success": "true",
            "gen_ai.request.model": "gpt-4o"
        },
        "Duration": 10000000
    },
    {
        "Timestamp": "2025-12-22 18:37:24.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_003",
        "ParentSpanId": "span_002",
        "SpanName": "llm.call",
        "SpanKind": "Internal",
        "ServiceName": "corto.worker",
        "SpanAttributes": {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.response.model": "gpt-4o",
            "gen_ai.prompt.0.content": "What is the weather?",
            "gen_ai.completion.0.content": "The weather is sunny."
        },
        "Duration": 5000000
    },
    {
        "Timestamp": "2025-12-22 18:37:25.545847221",
        "TraceId": "162b29522a339e6b1acb21b8041dcda5",
        "SpanId": "span_004",
        "ParentSpanId": "span_003",
        "SpanName": "tool.execution",
        "SpanKind": "Internal",
        "ServiceName": "corto.worker",
        "SpanAttributes": {
            "gen_ai.request.model": "gpt-4o",
            "gen_ai.completion.0.tool_calls.0.name": "weather_lookup",
            "gen_ai.completion.0.tool_calls.0.arguments": "{\"location\": \"NYC\"}"
        },
        "Duration": 2000000
    }
]


@pytest.fixture
def sample_otel_records() -> List[Dict[str, Any]]:
    """Provide sample OTel records for testing."""
    return SAMPLE_OTEL_RECORDS.copy()


@pytest.fixture
def extraction_service() -> TelemetryExtractionService:
    """Create a TelemetryExtractionService without Azure OpenAI (basic mode)."""
    return TelemetryExtractionService(
        azure_endpoint=None,
        azure_api_key=None
    )


@pytest.fixture
def knowledge_processor_with_dedup() -> KnowledgeProcessor:
    """Create a KnowledgeProcessor with dedup enabled."""
    return KnowledgeProcessor(
        enable_embeddings=False,  # Disable for faster tests
        enable_dedup=True,
        similarity_threshold=0.95
    )


@pytest.fixture
def knowledge_processor_no_dedup() -> KnowledgeProcessor:
    """Create a KnowledgeProcessor with dedup disabled."""
    return KnowledgeProcessor(
        enable_embeddings=False,
        enable_dedup=False,
        similarity_threshold=0.95
    )


@pytest.fixture
def data_repository() -> MockDataRepository:
    """Create a MockDataRepository instance."""
    return MockDataRepository()


@pytest.fixture
def sample_extraction_result(extraction_service, sample_otel_records) -> Dict[str, Any]:
    """Provide a sample extraction result."""
    return extraction_service.extract_entities_and_relations(sample_otel_records)

