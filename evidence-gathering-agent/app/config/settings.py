import os


class Settings:
    # Use TKF Data Layer when set (e.g. http://localhost:8088); otherwise mock repo
    DATA_LAYER_BASE_URL: str | None = os.getenv("TKF_DATA_LAYER_BASE_URL") or os.getenv("DATA_LAYER_BASE_URL")
    # Optional: caching layer for similar-concept search when no vector DB is hooked (e.g. http://localhost:8091)
    CACHING_LAYER_BASE_URL: str | None = os.getenv("CACHING_LAYER_BASE_URL")
    # When True and CACHING_LAYER_BASE_URL is set, use cache + graph for similar (bypass repo vector search)
    USE_CACHE_FOR_SIMILAR: bool = os.getenv("USE_CACHE_FOR_SIMILAR", "").lower() in ("1", "true", "yes")
    AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT: str | None = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    EG_MAX_DEPTH: int = int(os.getenv("EG_MAX_DEPTH", "4"))
    EG_PATH_LIMIT: int = int(os.getenv("EG_PATH_LIMIT", "20"))


settings = Settings()

