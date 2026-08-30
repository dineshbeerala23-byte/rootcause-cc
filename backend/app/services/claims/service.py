# ============================================================
# app/services/claims/service.py  (NEW)
# ============================================================
"""
Orchestrates evidence-grounded claim generation and verification.

Structurally parallel to app/services/reasoning/service.py. Depends only
on a ClaimGenerator and a ClaimVerifier -- never on EvidenceRetrievalService,
TopologyProvider, or the dataset-adapter layer directly. Callers are
responsible for supplying the RCACandidateResult, EvidenceBundle, and
ServiceGraph already produced by the reasoning layer.

Method-agnostic with respect to how the upstream RCACandidateResult was
produced: this service runs identically whether the candidate result came
from "deterministic" or "llm-assisted" reasoning mode, since claim
generation and verification only depend on the candidate result's
contents, not on how it was produced.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.claims import ClaimVerificationResult
from app.schemas.evidence import EvidenceBundle
from app.schemas.graph import ServiceGraph
from app.schemas.reasoning import RCACandidateResult
from app.services.claims.base import ClaimGenerator, ClaimVerifier
from app.services.claims.verifier import DeterministicClaimVerifier


class ClaimVerificationService:
    def __init__(
        self,
        claim_generator: ClaimGenerator,
        claim_verifier: Optional[ClaimVerifier] = None,
    ):
        self._claim_generator = claim_generator
        self._claim_verifier = claim_verifier or DeterministicClaimVerifier()

    async def analyze(
        self,
        candidate_result: RCACandidateResult,
        evidence: EvidenceBundle,
        graph: ServiceGraph,
    ) -> ClaimVerificationResult:
        claim_result = await self._claim_generator.generate(candidate_result, evidence, graph)
        return self._claim_verifier.verify(claim_result, evidence)
