"""Shared pytest fixtures for the caching layer."""
import pytest

from app.agent.caching_layer import CachingLayer


@pytest.fixture
def caching_layer() -> CachingLayer:
    """Provide a small CachingLayer instance suitable for unit tests."""
    return CachingLayer(vector_dimension=8)
