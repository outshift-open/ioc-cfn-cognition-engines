"""
Environment configuration using Pydantic Settings.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Service configuration
    service_name: str = Field(default="telemetry-extraction-service", env="SERVICE_NAME")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8086, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Azure OpenAI configuration
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str = Field(default="gpt-4o", env="AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-08-01-preview", env="AZURE_OPENAI_API_VERSION")
    
    # Knowledge processing configuration
    enable_embeddings: bool = Field(default=True, env="ENABLE_EMBEDDINGS")
    enable_dedup: bool = Field(default=True, env="ENABLE_DEDUP")
    similarity_threshold: float = Field(default=0.95, env="SIMILARITY_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton settings instance
settings = Settings()

