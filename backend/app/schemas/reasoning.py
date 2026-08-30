# ============================================================
# app/schemas/reasoning.py  (NEW)
# ============================================================
"""
Output schemas for the RCA Candidate Reasoning layer.

Deliberately excludes any dependency on the ground-truth schema module and
carries no field that represents "the answer" -- see CandidateCause. `service` is a
ranked guess, not a verified root cause, and `rationale` (when present) is
LLM-authored explanation text, not evaluation output.

as_of is always copied verbatim from the EvidenceBundle that produced a
result -- this schema does not recompute or widen a cutoff.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ReasoningMethod = Literal["deterministic", "llm-assisted"]


class CandidateCause(BaseModel):
    """
    One ranked candidate service. Intentionally has no fault-type-like
    field in this milestone -- see handoff notes on why that was deferred.
    """

    model_config = ConfigDict(extra="forbid")

    service: str
    score: float = Field(
        ..., description="Deterministic ranking score -- sum of supporting evidence scores."
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="EvidenceItem.column_name values that contributed to this candidate's score.",
    )
    graph_support: Optional[List[str]] = Field(
        default=None,
        description=(
            "Direct topology neighbors of this service in the case's ServiceGraph. "
            "None means graph adjacency was not evaluated for this system (e.g. the "
            "graph has no edges); an empty list means evaluated but no neighbors found."
        ),
    )
    rationale: Optional[str] = Field(
        default=None,
        description="LLM-authored explanation, added only in llm-assisted mode. Never ranks or filters candidates.",
    )


class RCACandidateResult(BaseModel):
    """
    Full output of one reasoning run for one case. `candidates` is always
    ordered by the deterministic ranking -- llm-assisted mode may only add
    `rationale` text to existing entries, never reorder, add, or remove them.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    as_of: datetime
    method: ReasoningMethod
    candidates: List[CandidateCause] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)