"""
Aggregates all v1 endpoint routers into a single APIRouter.

Later phases add their routers here (e.g. graph query) — this is the one
file that needs a one-line addition per new endpoint group.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import datasets, extraction, health, rca

api_router = APIRouter()

api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(extraction.router, prefix="/extraction", tags=["Extraction"])
api_router.include_router(rca.router, prefix="/rca", tags=["RCA"])