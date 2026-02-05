"""
Integration tests / API smoke tests for the FastAPI application.
"""
import pytest
import json
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_otel_payload():
    """Sample OTel payload for testing batch endpoint."""
    return [
        {
            "TraceId": "test_trace_001",
            "SpanId": "span_001",
            "ParentSpanId": "",
            "SpanName": "test.agent",
            "ServiceName": "test.service",
            "SpanAttributes": {
                "agent_id": "test_agent",
                "gen_ai.request.model": "gpt-4o"
            },
            "Duration": 1000000
        },
        {
            "TraceId": "test_trace_001",
            "SpanId": "span_002",
            "ParentSpanId": "span_001",
            "SpanName": "child.agent",
            "ServiceName": "test.service",
            "SpanAttributes": {
                "agent_id": "child_agent",
                "gen_ai.request.model": "gpt-4o"
            },
            "Duration": 500000
        }
    ]


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_service_info(self, client):
        """Test that root endpoint returns service info."""
        response = client.get("/")
        data = response.json()
        
        assert "message" in data
        assert "status" in data
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self, client):
        """Test that health endpoint returns health status."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "metrics" in data


class TestMetricsEndpoint:
    """Tests for the metrics endpoint."""
    
    def test_metrics_returns_200(self, client):
        """Test that metrics endpoint returns 200."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
    
    def test_metrics_returns_expected_fields(self, client):
        """Test that metrics endpoint returns expected fields."""
        response = client.get("/api/v1/metrics")
        data = response.json()
        
        assert "records_processed" in data
        assert "records_sent" in data
        assert "records_failed" in data
        assert "recent_errors" in data


class TestBatchExtractionEndpoint:
    """Tests for the batch extraction endpoint."""
    
    def test_batch_extraction_returns_200(self, client, sample_otel_payload):
        """Test that batch extraction endpoint returns 200."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content=json.dumps(sample_otel_payload),
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200
    
    def test_batch_extraction_returns_concepts(self, client, sample_otel_payload):
        """Test that batch extraction returns concepts."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content=json.dumps(sample_otel_payload),
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        
        assert "concepts" in data
        assert len(data["concepts"]) > 0
    
    def test_batch_extraction_returns_relations(self, client, sample_otel_payload):
        """Test that batch extraction returns relations."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content=json.dumps(sample_otel_payload),
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        
        assert "relations" in data
    
    def test_batch_extraction_returns_meta(self, client, sample_otel_payload):
        """Test that batch extraction returns metadata."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content=json.dumps(sample_otel_payload),
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        
        assert "meta" in data
        assert data["meta"]["records_processed"] == 2
    
    def test_batch_extraction_returns_request_id(self, client, sample_otel_payload):
        """Test that batch extraction returns knowledge_cognition_request_id."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content=json.dumps(sample_otel_payload),
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        
        assert "knowledge_cognition_request_id" in data
    
    def test_batch_extraction_extracts_agents(self, client, sample_otel_payload):
        """Test that batch extraction correctly extracts agents."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content=json.dumps(sample_otel_payload),
            headers={"Content-Type": "application/json"}
        )
        data = response.json()
        
        agent_names = [c["name"] for c in data["concepts"] if c["attributes"].get("concept_type") == "agent"]
        assert "test_agent" in agent_names
        assert "child_agent" in agent_names
    
    def test_batch_extraction_empty_body_returns_400(self, client):
        """Test that empty body returns 400."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content="[]",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
    
    def test_batch_extraction_invalid_json_returns_400(self, client):
        """Test that invalid JSON returns 400."""
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
    
    def test_batch_extraction_ndjson_format(self, client):
        """Test that NDJSON format is supported."""
        ndjson_payload = '{"TraceId": "t1", "SpanId": "s1", "SpanAttributes": {"agent_id": "agent1"}}\n{"TraceId": "t1", "SpanId": "s2", "SpanAttributes": {"agent_id": "agent2"}}'
        
        response = client.post(
            "/api/v1/extract/entities_and_relations/batch",
            content=ndjson_payload,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["meta"]["records_processed"] == 2


class TestFileExtractionEndpoint:
    """Tests for the file extraction endpoint."""
    
    def test_file_extraction_missing_file_returns_404(self, client):
        """Test that missing file returns 404."""
        response = client.get(
            "/api/v1/extract/entities_and_relations/from_file",
            params={"file_path": "/nonexistent/file.json"}
        )
        assert response.status_code == 404
    
    def test_file_extraction_invalid_extension_returns_400(self, client, tmp_path):
        """Test that invalid file extension returns 400."""
        # Create a temp file with invalid extension
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("[]")
        
        response = client.get(
            "/api/v1/extract/entities_and_relations/from_file",
            params={"file_path": str(invalid_file)}
        )
        assert response.status_code == 400

