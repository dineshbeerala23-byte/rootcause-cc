# ============================================================
# app/services/reasoning/base.py  (NEW)
# ============================================================
"""
Dataset/system-agnostic interface for RCA candidate generation.

Implementations must consume only EvidenceBundle + ServiceGraph +
NormalizedIncident. This interface intentionally has no dependency on the
dataset-adapter layer or the ground-truth schema -- a CandidateGenerator has
no way to reach either, by construction, regardless of what the caller
(RCAReasoningService) holds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.evidence import EvidenceBundle
from app.schemas.graph import ServiceGraph
from app.schemas.incident import NormalizedIncident
from app.schemas.reasoning import RCACandidateResult


class CandidateGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        incident: NormalizedIncident,
        evidence: EvidenceBundle,
        graph: ServiceGraph,
    ) -> RCACandidateResult:
        raise NotImplementedError