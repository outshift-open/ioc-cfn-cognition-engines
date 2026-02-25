from .data.mock_repo import MockDataRepository
from .data.http_repo import HttpDataRepository
from .data.cache_client import CacheClient
from .config.settings import settings


def get_repository():
    if settings.DATA_LAYER_BASE_URL:
        return HttpDataRepository(base_url=settings.DATA_LAYER_BASE_URL)
    return MockDataRepository()


def get_cache_client():
    """Return a cache client when CACHING_LAYER_BASE_URL is set; else None (bypass cache)."""
    if settings.CACHING_LAYER_BASE_URL:
        return CacheClient(base_url=settings.CACHING_LAYER_BASE_URL)
    return None


