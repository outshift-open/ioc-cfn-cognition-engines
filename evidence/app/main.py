# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from .api.routes import router as api_router
from .config.settings import settings


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s...", settings.service_name)
    yield
    logger.info("Shutting down %s...", settings.service_name)


def get_app() -> FastAPI:
    app = FastAPI(
        title="Evidence Gathering Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix="/api/knowledge-mgmt")

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


app = get_app()
