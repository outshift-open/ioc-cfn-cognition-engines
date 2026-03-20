import os


class Settings:
    # Use mocked DB (data layer) when set (e.g. http://localhost:8088); otherwise in-process mock repo
    DATA_LAYER_BASE_URL: str | None = os.getenv("MOCKED_DB_BASE_URL") or os.getenv("DATA_LAYER_BASE_URL")
    AZURE_OPENAI_ENDPOINT: str | None = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT: str | None = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    EG_MAX_DEPTH: int = int(os.getenv("EG_MAX_DEPTH", "4"))
    EG_PATH_LIMIT: int = int(os.getenv("EG_PATH_LIMIT", "20"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    service_name: str = "Evidence Gathering Agent"


settings = Settings()

