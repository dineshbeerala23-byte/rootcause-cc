# ============================================================
# app/schemas/evaluation.py  (NEW)
# ============================================================
"""
Evaluation Foundation schemas.

Service-only root-cause correctness: predicted_rank / top_k are computed
against GroundTruth.root_cause_service only. fault_type is carried
through for reporting only and never affects correctness -- per approved
design.

Carries no dependency the other way: this module does not import
DatasetAdapter and is not itself allowed to call get_ground_truth() --
that stays confined to app/services/evaluation/service.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CaseEvaluationResult(BaseModel):
    """
    Evaluation metrics for a single case, computed purely from a
    FinalRCAResult + GroundTruth pair (grounding counts come from
    FinalRCAResult.candidates[*].{supporting,contradicting,uncertain}_claim_ids,
    which already fold in ClaimVerificationResult).
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    system: str
    fault_type: str = Field(description="Reporting-only; never used to compute correctness.")
    root_cause_service: str = Field(description="Ground truth answer, service-only.")

    predicted_rank: Optional[int] = Field(
        default=None,
        description=(
            "1-indexed rank of root_cause_service within FinalRCAResult.candidates "
            "(which preserves RCACandidateResult's deterministic ranking -- never "
            "re-ranked by evaluation code). None if root_cause_service does not "
            "appear among candidates."
        ),
    )
    top_1_accuracy: bool
    top_3_accuracy: bool
    top_5_accuracy: bool

    grounding_precision: Optional[float] = Field(
        default=None,
        description=(
            "Of claims about root_cause_service that reached a definitive verdict "
            "(supported or unsupported), the fraction that were supported. None if "
            "no claim about root_cause_service reached a definitive verdict."
        ),
    )
    grounding_recall: Optional[float] = Field(
        default=None,
        description=(
            "Of all claims made about root_cause_service (supported + unsupported "
            "+ uncertain), the fraction that were supported. None if no claim was "
            "made about root_cause_service at all."
        ),
    )

    notes: List[str] = Field(default_factory=list)


class SystemEvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str
    case_count: int
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mean_grounding_precision: Optional[float] = None
    mean_grounding_recall: Optional[float] = None


class OverallEvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_count: int
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mean_grounding_precision: Optional[float] = None
    mean_grounding_recall: Optional[float] = None


class EvaluationRunResult(BaseModel):
    """
    Full output of one evaluation run. `notes` here is run-level (e.g.
    cases skipped because ground truth or the pipeline itself failed) --
    distinct from each CaseEvaluationResult's own per-case `notes`.
    """

    model_config = ConfigDict(extra="forbid")

    per_case: List[CaseEvaluationResult] = Field(default_factory=list)
    overall: OverallEvaluationSummary
    by_system: Dict[str, SystemEvaluationSummary] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)