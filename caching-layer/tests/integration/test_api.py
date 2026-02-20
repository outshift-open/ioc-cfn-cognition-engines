"""Integration tests for the caching layer FastAPI app."""
import pytest


class TestRootEndpoints:
    def test_root_returns_message(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_returns_status(self, client):
        response = client.get("/health")
        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "healthy"


class TestCacheRouter:
    @pytest.fixture
    def payload(self):
        return {"keys": ["alpha", "beta"], "metadata": {"ttl_seconds": 90}}

    def test_status_endpoint_returns_defaults(self, client):
        response = client.get("/api/v1/cache/status")
        body = response.json()
        assert response.status_code == 200
        assert "namespace" in body
        assert body["primed_requests"] == 0

    def test_prime_endpoint_updates_metrics(self, client, payload):
        response = client.post("/api/v1/cache/prime", json=payload)
        body = response.json()
        assert response.status_code == 200
        assert body["primed_keys"] == payload["keys"]

        status = client.get("/api/v1/cache/status").json()
        assert status["primed_requests"] == 1
        assert status["tracked_keys"] == payload["keys"]

    def test_store_endpoint_persists_text(self, client):
        response = client.post("/api/v1/cache/store", json={"text": "integration-doc"})
        body = response.json()

        assert response.status_code == 200
        assert "id" in body
        assert body["ntotal"] >= 1

    def test_search_endpoint_finds_recent_entry(self, client):
        unique_text = "integration-search-text"
        client.post("/api/v1/cache/store", json={"text": unique_text})

        response = client.post("/api/v1/cache/search", json={"text": unique_text, "k": 1})
        body = response.json()

        assert response.status_code == 200
        assert body["results"], "expected at least one match"
        assert body["results"][0]["text"] == unique_text
