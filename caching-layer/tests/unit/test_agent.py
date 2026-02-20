"""Unit tests for the CacheOrchestratorService."""
from app.agent.service import CacheOrchestratorService


class TestCacheOrchestratorService:
    def test_status_reflects_prime_count(self):
        service = CacheOrchestratorService(namespace="unit", default_ttl_seconds=30)
        service.prime_cache(keys=["unit:key"], metadata={"ttl_seconds": 45})
        status = service.get_cache_status()

        assert status["namespace"] == "unit"
        assert status["primed_requests"] == 1
        assert status["tracked_keys"] == ["unit:key"]

    def test_prime_uses_synthetic_key_when_empty(self):
        service = CacheOrchestratorService(namespace="unit", default_ttl_seconds=30)
        result = service.prime_cache()

        assert result["primed_keys"], "should generate placeholder key"
        assert result["ttl_seconds"] == 30

    def test_health_payload_contains_metrics(self):
        service = CacheOrchestratorService(namespace="unit", default_ttl_seconds=30)
        service.prime_cache(keys=["a", "b"], metadata={})

        health = service.report_health_and_diagnostics()

        assert health["status"] == "healthy"
        assert health["metrics"]["primed_requests"] == 1
        assert "recent_errors" in health
