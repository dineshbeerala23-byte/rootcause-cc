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
from app.services.claims.llm_generator import LLMClaimGenerator
from app.services.claims.service import ClaimVerificationService
from app.services.datasets.base import DatasetAdapter
from app.services.datasets.rcaeval.adapter import RCAEvalAdapter
from app.services.evidence.service import EvidenceRetrievalService
from app.services.extraction_service import ExtractionService
from app.services.final_result.service import FinalResultAssemblyService
from app.services.graph.base import TopologyProvider
from app.services.graph.provider import StaticTopologyProvider
from app.services.llm.base import LLMProvider
from app.services.llm.groq_provider import GroqProvider
from app.services.pipeline.service import RCAPipelineService
from app.services.reasoning.llm_reasoner import LLMRationaleAnnotator
from app.services.reasoning.service import RCAReasoningService


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


# ------------------------------------------------------------------
# RCA pipeline dependencies (NEW -- Evidence-Grounded RCA Pipeline
# API/CLI wiring milestone). Additive only; nothing above this
# comment was changed.
# ------------------------------------------------------------------


def get_evidence_service(
    adapter: DatasetAdapter = Depends(get_rcaeval_adapter),
) -> EvidenceRetrievalService:
    return EvidenceRetrievalService(adapter)


def get_topology_provider() -> TopologyProvider:
    return StaticTopologyProvider()


def get_reasoning_service(
    evidence_service: EvidenceRetrievalService = Depends(get_evidence_service),
    topology_provider: TopologyProvider = Depends(get_topology_provider),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> RCAReasoningService:
    return RCAReasoningService(
        evidence_service,
        topology_provider,
        rationale_annotator=LLMRationaleAnnotator(llm_provider),
    )


def get_claims_service(
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> ClaimVerificationService:
    return ClaimVerificationService(claim_generator=LLMClaimGenerator(llm_provider))


def get_final_result_service() -> FinalResultAssemblyService:
    return FinalResultAssemblyService()


def get_pipeline_service(
    adapter: DatasetAdapter = Depends(get_rcaeval_adapter),
    evidence_service: EvidenceRetrievalService = Depends(get_evidence_service),
    topology_provider: TopologyProvider = Depends(get_topology_provider),
    reasoning_service: RCAReasoningService = Depends(get_reasoning_service),
    claims_service: ClaimVerificationService = Depends(get_claims_service),
    final_result_service: FinalResultAssemblyService = Depends(get_final_result_service),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> RCAPipelineService:
    return RCAPipelineService(
        adapter=adapter,
        evidence_service=evidence_service,
        topology_provider=topology_provider,
        reasoning_service=reasoning_service,
        claims_service=claims_service,
        final_result_service=final_result_service,
        llm_provider=llm_provider,
    )
