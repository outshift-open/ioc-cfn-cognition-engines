"""Shared pytest fixtures for the caching layer."""
import pytest
from fastapi.testclient import TestClient

from app.agent.service import CacheOrchestratorService
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def cache_service() -> CacheOrchestratorService:
    return CacheOrchestratorService(namespace="test", default_ttl_seconds=60)


@pytest.fixture
def sample_prime_request() -> dict:
    return {"keys": ["alpha", "beta"], "metadata": {"ttl_seconds": 120}}
