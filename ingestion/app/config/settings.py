# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""
Environment configuration using Pydantic Settings.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Resolve .env from repo root (one level above ingestion/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore env vars for other services (e.g. MOCKED_DB_BASE_URL, NEGOTIATOR_STRATEGY)
    )

    # Service configuration
    service_name: str = Field(default="telemetry-extraction-service")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8086)
    log_level: str = Field(default="INFO")

    # Azure OpenAI configuration
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_api_key: Optional[str] = Field(default=None)
    azure_openai_deployment: str = Field(default="gpt-4o")
    azure_openai_api_version: str = Field(default="2024-08-01-preview")

    # Knowledge processing configuration
    enable_embeddings: bool = Field(default=True)
    enable_dedup: bool = Field(default=True)
    enable_rag_ingest: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.95)
    # Local path to granite-embedding-30m-english (overrides Hugging Face when set)
    embedding_model_path: Optional[str] = Field(default=None)

    # FAISS vector store (in-process via caching-layer library)
    enable_faiss_storage: bool = Field(default=True)
    faiss_vector_dimension: int = Field(default=384)
    faiss_metric: str = Field(default="l2")


# Singleton settings instance
settings = Settings()
