from fastapi import Request

from .data.mock_repo import MockDataRepository
from .data.http_repo import HttpDataRepository
from .data.cache_client import CacheClient
from .config.settings import settings


def get_repository(request: Request):
    # Unified app: gateway passes in-memory cache_layer; no external graph DB is running.
    # Use mock repo so neighbors_by_name etc. do not trigger HTTP to DATA_LAYER_BASE_URL.
    if getattr(request.app.state, "cache_layer", None) is not None:
        return MockDataRepository()
    if settings.DATA_LAYER_BASE_URL:
        return HttpDataRepository(base_url=settings.DATA_LAYER_BASE_URL)
    return MockDataRepository()


def get_cache_client():
    """Return a cache client when CACHING_LAYER_BASE_URL is set; else None (bypass cache)."""
    if settings.CACHING_LAYER_BASE_URL:
        return CacheClient(base_url=settings.CACHING_LAYER_BASE_URL)
    return None


def get_cache_layer(request: Request):
    """Return the shared in-memory CachingLayer when running under the unified app (Option A)."""
    return getattr(request.app.state, "cache_layer", None)


