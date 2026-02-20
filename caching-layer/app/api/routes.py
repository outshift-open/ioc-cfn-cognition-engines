"""REST API routes for the caching layer."""
import logging

import numpy as np
from fastapi import APIRouter, Depends

from ..agent.caching_layer import CachingLayer
from ..agent.service import CacheOrchestratorService
from ..dependencies import get_cache_service, get_caching_layer
from .schemas import (
    CacheStatusResponse,
    PrimeCacheRequest,
    PrimeCacheResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StoreKnowledgeRequest,
    StoreKnowledgeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.get("/status", response_model=CacheStatusResponse)
async def cache_status(service: CacheOrchestratorService = Depends(get_cache_service)) -> CacheStatusResponse:
    """Return the latest cache status snapshot."""
    status = service.get_cache_status()
    logger.debug("Cache status requested: %s", status)
    return CacheStatusResponse(**status)


@router.post("/prime", response_model=PrimeCacheResponse)
async def prime_cache(
    payload: PrimeCacheRequest,
    service: CacheOrchestratorService = Depends(get_cache_service),
) -> PrimeCacheResponse:
    """Simulate cache priming by recording the requested keys."""
    result = service.prime_cache(keys=payload.keys, metadata=payload.metadata)
    logger.info("Prime request processed for %s keys", len(result["primed_keys"]))
    return PrimeCacheResponse(**result)


@router.post("/store", response_model=StoreKnowledgeResponse)
async def store_knowledge(
    payload: StoreKnowledgeRequest,
    layer: CachingLayer = Depends(get_caching_layer),
) -> StoreKnowledgeResponse:
    """Store caller-provided text or vector in the FAISS-backed cache."""
    vector = np.array(payload.vector, dtype="float32") if payload.vector is not None else None
    result = layer.store_knowledge(text=payload.text, vector=vector)
    logger.debug("Stored knowledge with id=%s", result["id"])
    return StoreKnowledgeResponse(**result)


@router.post("/search", response_model=SearchResponse)
async def search_similar(
    payload: SearchRequest,
    layer: CachingLayer = Depends(get_caching_layer),
) -> SearchResponse:
    """Search for the nearest cached payloads using text or raw vectors."""
    vector = np.array(payload.vector, dtype="float32") if payload.vector is not None else None
    matches = layer.search_similar(text=payload.text, vector=vector, k=payload.k)
    return SearchResponse(results=[SearchResult(**match) for match in matches])
