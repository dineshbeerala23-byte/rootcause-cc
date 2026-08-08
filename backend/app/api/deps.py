"""
Shared FastAPI dependency providers.

Kept separate from individual endpoint modules so services can be composed
from smaller pieces (Settings -> LLMProvider -> ExtractionService, or
Settings -> DatasetAdapter) and so tests can override a single dependency
(get_llm_provider, get_rcaeval_adapter) with a fake/fixture-backed double,
without needing to reach into endpoint internals.
"""

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.datasets.base import DatasetAdapter
from app.services.datasets.rcaeval.adapter import RCAEvalAdapter
from app.services.extraction_service import ExtractionService
from app.services.llm.base import LLMProvider
from app.services.llm.groq_provider import GroqProvider


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """
    Returns the configured LLM provider. Groq is the only provider today;
    swapping or adding providers later only requires changing this factory
    function, not any calling code.
    """
    return GroqProvider(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def get_extraction_service(
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> ExtractionService:
    return ExtractionService(llm_provider=llm_provider)


def get_rcaeval_adapter(settings: Settings = Depends(get_settings)) -> DatasetAdapter:
    """
    Returns the configured RCAEval dataset adapter. RCAEval is the only
    dataset integration today; adding a second dataset later only
    requires changing this factory function (or adding a sibling one),
    not any calling code.
    """
    return RCAEvalAdapter(settings=settings)