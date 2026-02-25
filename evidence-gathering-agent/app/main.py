from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from .api.routes import router as api_router


def get_app() -> FastAPI:
    app = FastAPI(title="Evidence Gathering Agent", version="0.1.0")
    app.include_router(api_router, prefix="/api/knowledge-mgmt")

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


app = get_app()

