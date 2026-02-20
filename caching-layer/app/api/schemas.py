"""Pydantic models for the caching layer API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PrimeCacheRequest(BaseModel):
    """Payload used to prime cache entries."""

    keys: Optional[List[str]] = Field(default=None, description="List of cache keys to prime")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata for experimentation")


class CacheStatusResponse(BaseModel):
    """Current cache orchestration status."""

    namespace: str
    default_ttl_seconds: int
    primed_requests: int
    tracked_keys: List[str]
    last_prime_timestamp: Optional[str]


class PrimeCacheResponse(BaseModel):
    """Response returned after simulating prime operations."""

    primed_keys: List[str]
    applied_namespace: str
    ttl_seconds: int
    metadata: Dict[str, Any]
    diagnostics: Dict[str, Any]


class StoreKnowledgeRequest(BaseModel):
    """Request payload for persisting vectors or text."""

    text: Optional[str] = None
    vector: Optional[List[float]] = None


class StoreKnowledgeResponse(BaseModel):
    """Response returned after storing knowledge."""

    id: int
    ntotal: int


class SearchRequest(BaseModel):
    """Request payload for similarity search."""

    text: Optional[str] = None
    vector: Optional[List[float]] = None
    k: int = Field(default=5, ge=1, description="Maximum number of matches to return")


class SearchResult(BaseModel):
    """Single similarity search hit."""

    id: int
    score: float
    text: str


class SearchResponse(BaseModel):
    """Similarity search response."""

    results: List[SearchResult]


class HealthResponse(BaseModel):
    """Health payload returned by /health."""

    status: str
    initialized: bool
    metrics: Dict[str, Any]
    recent_errors: List[str]
