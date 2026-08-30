# ============================================================
# app/api/v1/endpoints/rca.py  (NEW)
# ============================================================
"""
End-to-end RCA pipeline endpoint.

Thin by design, same convention as app/api/v1/endpoints/extraction.py:
request validation and response shaping only. All orchestration logic
lives in app.services.pipeline.service.RCAPipelineService. Errors raised
anywhere in the pipeline are RootCauseError subclasses and are handled by
the global exception handlers registered in app.main -- nothing here
needs its own try/except.
"""

import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_pipeline_service
from app.schemas.pipeline import RCAAnalysisRequest, RCAAnalysisResponse
from app.services.pipeline.service import RCAPipelineService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/{case_id}/analyze",
    response_model=RCAAnalysisResponse,
    summary="Run the end-to-end evidence-grounded RCA pipeline for one case",
    description=(
        "Resolves the given opaque case_id to a NormalizedIncident, retrieves "
        "evidence and topology, generates ranked root-cause candidates, "
        "generates and verifies evidence-grounded claims about them, and "
        "returns one final, case-scoped, evidence-grounded RCA result."
    ),
)
async def analyze_case(
    case_id: str,
    payload: RCAAnalysisRequest = RCAAnalysisRequest(),
    service: RCAPipelineService = Depends(get_pipeline_service),
) -> RCAAnalysisResponse:
    logger.info("RCA pipeline analysis requested for case_id=%s, mode=%s", case_id, payload.mode)
    response = await service.analyze(case_id, mode=payload.mode)
    logger.info(
        "RCA pipeline analysis complete for case_id=%s: %d candidates.",
        case_id,
        len(response.result.candidates),
    )
    return response
