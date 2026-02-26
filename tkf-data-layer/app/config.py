"""Configuration from environment."""
import os
from typing import Optional


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


class Settings:
    """TKF Data Layer settings."""

    def __init__(self) -> None:
        # Neo4j
        self.neo4j_uri: str = _env("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user: str = _env("NEO4J_USER", "neo4j")
        self.neo4j_password: str = _env("NEO4J_PASSWORD", "password")

        # Embeddings (optional). fastembed expects full id; normalize short name.
        _model = _env("TKF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        if _model.strip() == "all-MiniLM-L6-v2":
            _model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model_name: str = _model
        self.embedding_dimensions: int = int(_env("TKF_EMBEDDING_DIMENSIONS", "384"))
        self.embedding_batch_size: int = int(_env("TKF_EMBEDDING_BATCH_SIZE", "32"))

        # Server
        self.host: str = _env("TKF_DATA_LAYER_HOST", "0.0.0.0")
        self.port: int = int(_env("TKF_DATA_LAYER_PORT", "8088"))

        # KCR data path (for load script)
        kcr = _env("KCR_JSON_PATH", "")
        self.kcr_json_path: Optional[str] = kcr if kcr else None


def get_settings() -> Settings:
    return Settings()
