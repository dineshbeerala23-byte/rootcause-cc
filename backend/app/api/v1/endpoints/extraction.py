"""
Entity/relationship extraction endpoint (Phase 2).

Thin by design: request validation and response shaping only. All prompting
and parsing logic lives in app.services.extraction_service; the LLM call
itself lives in app.services.llm.groq_provider. Errors raised by either are
RootCauseError subclasses and are handled by the global exception handlers
registered in app.main — nothing here needs its own try/except.
"""

import logging

from fastapi import APIRouter, Depends

from app.api.deps import get_extraction_service
from app.schemas.extraction import ExtractionRequest, ExtractionResponse
from app.services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=ExtractionResponse,
    summary="Extract structured entities and relationships from incident/system text",
    description=(
        "Sends the given text to the configured LLM provider and returns validated, "
        "structured entities and relationships suitable for later knowledge-graph "
        "construction (Phase 3)."
    ),
)
async def extract_entities(
    payload: ExtractionRequest,
    service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionResponse:
    logger.info("Extraction requested for %d characters of text.", len(payload.text))
    result = await service.extract(payload.text)
    logger.info(
        "Extraction complete: %d entities, %d relationships, %d warnings.",
        len(result.entities),
        len(result.relationships),
        len(result.warnings),
    )
    return ExtractionResponse(
        provider=service.provider_name,
        model=service.model_name,
        result=result,
    )