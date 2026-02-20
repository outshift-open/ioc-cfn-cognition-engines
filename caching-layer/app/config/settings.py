"""Environment configuration for the caching layer."""
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from env vars."""

    service_name: str = Field(default="caching-layer-service", env="SERVICE_NAME")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8091, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    cache_namespace: str = Field(default="default", env="CACHE_NAMESPACE")
    default_cache_ttl_seconds: int = Field(default=300, env="DEFAULT_CACHE_TTL_SECONDS")
    cache_vector_dimension: int = Field(default=1536, env="CACHE_VECTOR_DIMENSION")
    cache_metric: str = Field(default="l2", env="CACHE_METRIC")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
