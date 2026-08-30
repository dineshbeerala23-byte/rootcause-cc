# ============================================================
# app/services/reasoning/service.py  (NEW)
# ============================================================
"""
Orchestrates RCA candidate generation.

Depends only on EvidenceRetrievalService and TopologyProvider -- never on
the dataset-adapter layer. This is a structural (not just conventional)
guarantee that the ground-truth accessor is unreachable from this service:
there is no adapter reference here through which it could ever be called.

mode="llm-assisted" only ever ADDS rationale text to the deterministic
candidate set produced first; it never changes which candidates exist or
their order. If llm-assisted mode is requested but no rationale annotator
is configured, or the annotator fails for any reason, the deterministic
result is returned with an explanatory note instead of raising.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.evidence import EvidenceQuery
from app.schemas.incident import NormalizedIncident
from app.schemas.reasoning import RCACandidateResult
from app.services.evidence.service import EvidenceRetrievalService
from app.services.graph.base import TopologyProvider
from app.services.reasoning.base import CandidateGenerator
from app.services.reasoning.deterministic import DeterministicCandidateGenerator
from app.services.reasoning.llm_reasoner import LLMRationaleAnnotator


class RCAReasoningService:
    def __init__(
        self,
        evidence_service: EvidenceRetrievalService,
        topology_provider: TopologyProvider,
        candidate_generator: Optional[CandidateGenerator] = None,
        rationale_annotator: Optional[LLMRationaleAnnotator] = None,
    ):
        self._evidence_service = evidence_service
        self._topology_provider = topology_provider
        self._candidate_generator = candidate_generator or DeterministicCandidateGenerator()
        self._rationale_annotator = rationale_annotator

    async def analyze(
        self,
        incident: NormalizedIncident,
        mode: str = "deterministic",
        evidence_query: Optional[EvidenceQuery] = None,
    ) -> RCACandidateResult:
        if mode not in ("deterministic", "llm-assisted"):
            raise ValueError(f"Unknown reasoning mode: {mode!r}")

        query = evidence_query or EvidenceQuery()
        evidence = self._evidence_service.get_evidence(incident.case_id, query)
        graph = self._topology_provider.get_graph(incident.metadata.system)

        result = self._candidate_generator.generate(incident, evidence, graph)

        if mode == "llm-assisted":
            if self._rationale_annotator is None:
                return result.model_copy(
                    update={
                        "notes": [
                            *result.notes,
                            "llm-assisted mode requested but no rationale annotator "
                            "configured; returning deterministic result.",
                        ]
                    }
                )
            result = await self._rationale_annotator.annotate(result, evidence, graph)

        return result