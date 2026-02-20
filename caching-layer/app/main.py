"""FastAPI entrypoint for the caching layer."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api.routes import router as cache_router
from .config.settings import settings
from .dependencies import get_cache_service, get_caching_layer

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("caching-layer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.service_name)
    # Ensure caching layer is initialized at startup for predictable FAISS configuration.
    get_caching_layer()
    yield
    logger.info("Stopping %s", settings.service_name)


app = FastAPI(
    title="Caching Layer Service",
    description="Skeleton API for the caching infrastructure",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(cache_router)


@app.get("/", tags=["root"])
async def root() -> dict:
    return {"message": "Caching layer is online", "service": settings.service_name}


@app.get("/health", tags=["health"])
async def health_check() -> JSONResponse:
    diagnostics = get_cache_service().report_health_and_diagnostics()
    return JSONResponse(content=diagnostics)


def run_server() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run_server()
