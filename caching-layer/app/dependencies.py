"""Dependency factories for the caching layer."""
from functools import lru_cache

from .agent.caching_layer import CachingLayer
from .agent.service import CacheOrchestratorService
from .config.settings import settings
from .data.base import InMemoryCacheStore


@lru_cache()
def get_cache_service() -> CacheOrchestratorService:
    """Return a singleton cache orchestrator instance."""
    return CacheOrchestratorService(
        namespace=settings.cache_namespace,
        default_ttl_seconds=settings.default_cache_ttl_seconds,
    )


def get_cache_store() -> InMemoryCacheStore:
    """Return a throwaway in-memory cache store for experimentation."""
    return InMemoryCacheStore()


@lru_cache()
def get_caching_layer() -> CachingLayer:
    """Provide a singleton FAISS-backed caching layer."""
    return CachingLayer(
        vector_dimension=settings.cache_vector_dimension,
        metric=settings.cache_metric,
    )
