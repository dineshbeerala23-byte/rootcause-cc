# ============================================================
# app/schemas/final_result.py  (NEW)
# ============================================================
"""
Final Evidence-Grounded RCA Result Assembly schemas.

Combines an RCACandidateResult with a ClaimVerificationResult into one
case-scoped result: each original candidate, in its original order,
annotated with a deterministic grounding_status derived from that
candidate's own claim verdicts.

This is an internal aggregation layer only (per approved scope) -- not
an API/CLI response shape. It carries no ground-truth fields and no
dependency on app/schemas/ground_truth.py.

candidate_grounding rule (deterministic, no scores/thresholds):
  - zero claims for this candidate           -> "uncertain"
  - any claim verdict is "unsupported"       -> "unsupported"
  - all claim verdicts are "supported"       -> "supported"
  - otherwise (mixed supported/uncertain)    -> "uncertain"

Reuses the same three-value vocabulary as ClaimVerificationStatus,
applied at candidate scope instead of claim scope.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.reasoning import CandidateCause

CandidateGroundingStatus = Literal["supported", "unsupported", "uncertain"]


class GroundedCandidate(BaseModel):
    """
    One candidate from the original RCACandidateResult, unchanged, plus
    its aggregated grounding status and the claim ids that produced it.
    """

    model_config = ConfigDict(extra="forbid")

    candidate: CandidateCause = Field(
        ..., description="Copied verbatim from RCACandidateResult.candidates -- never modified."
    )
    grounding_status: CandidateGroundingStatus
    supporting_claim_ids: List[str] = Field(
        default_factory=list,
        description="claim_id values whose verdict for this candidate was 'supported'.",
    )
    contradicting_claim_ids: List[str] = Field(
        default_factory=list,
        description="claim_id values whose verdict for this candidate was 'unsupported'.",
    )
    uncertain_claim_ids: List[str] = Field(
        default_factory=list,
        description="claim_id values whose verdict for this candidate was 'uncertain'.",
    )


class FinalRCAResult(BaseModel):
    """
    Full output of the final assembly step for one case.

    `candidates` preserves the exact order of the originating
    RCACandidateResult.candidates -- this assembly step never re-ranks.

    `as_of` is copied verbatim from the input ClaimVerificationResult
    (which itself copies verbatim from the EvidenceBundle throughout the
    pipeline) -- never recomputed.

    `notes` is cumulative: RCACandidateResult.notes +
    ClaimVerificationResult.notes (which already includes
    ClaimGenerationResult.notes) + this assembly step's own notes
    (e.g. a case_id or as_of mismatch between inputs).
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    as_of: datetime
    candidates: List[GroundedCandidate] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)