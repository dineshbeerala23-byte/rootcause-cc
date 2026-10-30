"""
RootCause-CC â€” application entrypoint.

Run with:
    uvicorn app.main:app --reload

This module wires together configuration, logging, middleware, exception
handlers, and versioned API routing. It intentionally contains no business
logic â€” that belongs in later phases' modules under app/, imported and
mounted here.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import (
    RootCauseError,
    http_exception_handler,
    rootcause_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.app_logging import setup_logging

settings = get_settings()
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown hook.

    Later phases will initialize long-lived clients here (Neo4j driver,
    pgvector connection pool, Groq client), attach them to app.state, and
    close them cleanly on shutdown. Nothing to initialize yet in Phase 1.
    """
    logger.info("Starting %s [%s] (v%s)", settings.APP_NAME, settings.APP_ENV, settings.APP_VERSION)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        description=(
            "GraphRAG-powered incident intelligence platform with "
            "corroboration-weighted, calibration-learned root-cause reasoning."
        ),
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.add_exception_handler(RootCauseError, rootcause_exception_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    application.include_router(api_router, prefix=settings.API_V1_PREFIX)
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    frontend_dir = frontend_dir.resolve()

    application.mount(
        "/ui",
        StaticFiles(directory=str(frontend_dir), html=True),
        name="frontend",
    )

    @application.get("/", tags=["Root"], summary="API root")
    async def root() -> dict:
        return {
            "service": settings.APP_NAME,
            "status": "running",
            "environment": settings.APP_ENV,
            "docs": "/docs",
            "api_version": "v1",
            "health_check": f"{settings.API_V1_PREFIX}/health",
        }

    return application


app = create_app()
