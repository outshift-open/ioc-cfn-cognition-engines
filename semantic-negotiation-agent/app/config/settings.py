"""
Environment configuration using Pydantic Settings.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Resolve .env relative to the project root (semantic-negotiation-agent/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Service configuration
    service_name: str = Field(default="semantic-negotiation-agent")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8089)
    log_level: str = Field(default="INFO")

    # Negotiation defaults
    negotiation_n_steps: int = Field(
        default=100,
        description="Default maximum SAO rounds per negotiation session",
    )

    negotiator_strategy: str = Field(
        default="BoulwareTBNegotiator",
        description=(
            "NegMAS SAO negotiator class name. "
            "Available: BoulwareTBNegotiator, ConcederTBNegotiator, LinearTBNegotiator, "
            "NaiveTitForTatNegotiator, SimpleTitForTatNegotiator, "
            "AspirationNegotiator, CABNegotiator, CANNegotiator, "
            "ToughNegotiator, NiceNegotiator, MiCRONegotiator."
        ),
    )


# Singleton settings instance
settings = Settings()
