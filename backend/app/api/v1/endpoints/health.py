"""
Health check endpoint.

Reports application liveness plus the configuration status of downstream
services. neo4j_graph_db and vector_store remain "not_configured" — they
are planned for Phase 3 and this endpoint intentionally does not change
their status yet. llm_provider_groq now reflects real configuration state
(Phase 2): "ok" once GROQ_API_KEY is set, "not_configured" otherwise. This
is a configuration check only, not a live ping against the Groq API.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.health import ComponentStatus, HealthResponse

router = APIRouter()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Detailed service health check",
)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    groq_configured = settings.groq_configured

    components = [
        ComponentStatus(
            name="neo4j_graph_db",
            status="not_configured",
            detail="Planned for Phase 3 (knowledge graph construction).",
        ),
        ComponentStatus(
            name="vector_store",
            status="not_configured",
            detail="Planned for Phase 3 (pgvector / Supabase).",
        ),
        ComponentStatus(
            name="llm_provider_groq",
            status="ok" if groq_configured else "not_configured",
            detail=(
                f"Configured to use model '{settings.GROQ_MODEL}'."
                if groq_configured
                else "Set GROQ_API_KEY to enable entity/relationship extraction."
            ),
        ),
    ]

    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )