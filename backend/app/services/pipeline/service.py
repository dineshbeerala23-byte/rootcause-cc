# ============================================================
# app/services/pipeline/service.py  (NEW)
# ============================================================
"""
End-to-end RCA pipeline orchestration: case_id -> RCAAnalysisResponse.

This is the single place the full call sequence (incident -> evidence ->
graph -> reasoning -> claims -> final result) is assembled, so the API
endpoint and the CLI command both call this instead of each re-deriving
the sequence independently.

Depends only on already-existing, unmodified services:
  DatasetAdapter, EvidenceRetrievalService, TopologyProvider,
  RCAReasoningService, ClaimVerificationService, FinalResultAssemblyService,
  LLMProvider (for provider/model metadata only).

Does not call GroundTruth or DatasetAdapter.get_ground_truth() anywhere.
Does not access raw telemetry directly -- only through EvidenceBundle via
EvidenceRetrievalService, exactly as the reasoning layer already does.

`mode` controls only RCAReasoningService's rationale-annotation behavior.
Claim generation is always LLM-dependent, per the approved design --
this is unaffected by `mode`.
"""

from __future__ import annotations

from app.schemas.evidence import EvidenceQuery
from app.schemas.pipeline import RCAAnalysisResponse
from app.schemas.reasoning import ReasoningMethod
from app.services.claims.service import ClaimVerificationService
from app.services.datasets.base import DatasetAdapter
from app.services.evidence.service import EvidenceRetrievalService
from app.services.final_result.service import FinalResultAssemblyService
from app.services.graph.base import TopologyProvider
from app.services.llm.base import LLMProvider
from app.services.reasoning.service import RCAReasoningService


class RCAPipelineService:
    def __init__(
        self,
        adapter: DatasetAdapter,
        evidence_service: EvidenceRetrievalService,
        topology_provider: TopologyProvider,
        reasoning_service: RCAReasoningService,
        claims_service: ClaimVerificationService,
        final_result_service: FinalResultAssemblyService,
        llm_provider: LLMProvider,
    ):
        self._adapter = adapter
        self._evidence_service = evidence_service
        self._topology_provider = topology_provider
        self._reasoning_service = reasoning_service
        self._claims_service = claims_service
        self._final_result_service = final_result_service
        self._llm_provider = llm_provider

    async def analyze(self, case_id: str, mode: ReasoningMethod = "deterministic") -> RCAAnalysisResponse:
        incident = self._adapter.get_incident(case_id)

        evidence_query = EvidenceQuery()
        evidence = self._evidence_service.get_evidence(incident.case_id, evidence_query)
        graph = self._topology_provider.get_graph(incident.metadata.system)

        candidate_result = await self._reasoning_service.analyze(
            incident, mode=mode, evidence_query=evidence_query
        )
        claim_verification_result = await self._claims_service.analyze(
            candidate_result, evidence, graph
        )
        final_result = self._final_result_service.assemble(
            candidate_result, claim_verification_result
        )

        return RCAAnalysisResponse(
            case_id=incident.case_id,
            mode=mode,
            provider=self._llm_provider.name,
            model=self._llm_provider.model,
            result=final_result,
        )
