# ============================================================
# app/services/claims/base.py  (NEW)
# ============================================================
"""
Abstract interfaces for the claim generation and verification layer.

Mirrors the ABC pattern used by app/services/reasoning/base.py's
CandidateGenerator: a narrow, swappable interface so
ClaimVerificationService can be tested against fakes and so alternative
implementations can be substituted without touching the orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.claims import ClaimGenerationResult, ClaimVerificationResult
from app.schemas.evidence import EvidenceBundle
from app.schemas.graph import ServiceGraph
from app.schemas.reasoning import RCACandidateResult


class ClaimGenerator(ABC):
    """
    Produces claims about an already-ranked RCACandidateResult.

    Implementations must not add, remove, or reorder the candidates
    themselves -- they only read from the candidate result to author
    claim text. Implementations must never cite evidence outside a
    candidate's own supporting_evidence.
    """

    @abstractmethod
    async def generate(
        self,
        candidate_result: RCACandidateResult,
        evidence: EvidenceBundle,
        graph: ServiceGraph,
    ) -> ClaimGenerationResult:
        raise NotImplementedError


class ClaimVerifier(ABC):
    """
    Independently checks each Claim's cited_evidence against an
    EvidenceBundle. Must not trust anything the ClaimGenerator asserted
    about its own claims -- recomputes matched/unmatched evidence from
    the bundle directly.
    """

    @abstractmethod
    def verify(
        self,
        claim_result: ClaimGenerationResult,
        evidence: EvidenceBundle,
    ) -> ClaimVerificationResult:
        raise NotImplementedError
