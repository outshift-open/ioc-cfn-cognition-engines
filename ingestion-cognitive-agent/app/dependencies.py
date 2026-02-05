"""
Dependency injection configuration.

This module provides factory functions for creating service instances
with the appropriate dependencies injected.
"""

from functools import lru_cache
from typing import Optional

from .config.settings import settings
from .agent.service import TelemetryExtractionService
from .agent.knowledge_processor import KnowledgeProcessor
from .data.mock_repo import MockDataRepository


@lru_cache()
def get_data_repository() -> MockDataRepository:
    """
    Get the data repository instance.
    
    Returns:
        MockDataRepository instance (can be swapped for other implementations)
    """
    return MockDataRepository()


@lru_cache()
def get_extraction_service() -> TelemetryExtractionService:
    """
    Get the telemetry extraction service instance.
    
    Returns:
        TelemetryExtractionService configured with Azure OpenAI settings
    """
    return TelemetryExtractionService(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_deployment,
        azure_api_version=settings.azure_openai_api_version
    )


def get_knowledge_processor() -> KnowledgeProcessor:
    """
    Get a knowledge processor instance.
    
    Note: Not cached as processor may be configured differently per request.
    
    Returns:
        KnowledgeProcessor configured with current settings
    """
    return KnowledgeProcessor(
        enable_embeddings=settings.enable_embeddings,
        enable_dedup=settings.enable_dedup,
        similarity_threshold=settings.similarity_threshold
    )

