"""TKF Data Layer: FastAPI app and lifecycle."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env from project root
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_env_path)

from .config import get_settings
from .graph.neo4j_store import Neo4jStore
from .semantic.embeddings import SemanticSearch
from .api.routes import router as api_router

_store: Optional[Neo4jStore] = None
_semantic: Optional[SemanticSearch] = None


def get_store() -> Optional[Neo4jStore]:
    return _store


def get_semantic() -> Optional[SemanticSearch]:
    return _semantic


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _semantic
    settings = get_settings()
    try:
        _store = Neo4jStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        await _store.connect()
        _semantic = SemanticSearch(
            store=_store,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
        )
        yield
    finally:
        if _store:
            await _store.close()
        _store = None
        _semantic = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="TKF Data Layer",
        version="0.1.0",
        description="Neo4j-backed graph API for evidence-gathering-agent",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
