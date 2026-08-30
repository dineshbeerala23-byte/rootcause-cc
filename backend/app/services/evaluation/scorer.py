# ============================================================
# app/services/evaluation/scorer.py  (NEW)
# ============================================================
"""
EvaluationScorer: pure, deterministic scoring.

Deliberately has NO dependency on DatasetAdapter and NEVER calls
get_ground_truth() -- it only accepts an already-fetched GroundTruth
object as an argument, same "orchestrator fetches, this layer combines"
split as FinalResultAssemblyService / ClaimVerificationService.

Takes case_id/system as plain strings rather than a full CaseMetadata
object -- this milestone doesn't need ModalityAvailability or any other
CaseMetadata field, so the scorer's input surface stays minimal.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.schemas.evaluation import (
    CaseEvaluationResult,
    OverallEvaluationSummary,
    SystemEvaluationSummary,
)
from app.schemas.final_result import FinalRCAResult
from app.schemas.ground_truth import GroundTruth


class EvaluationScorer:
    """Pure, deterministic. No adapter, no I/O, no ground-truth fetching."""

    def score_case(
        self,
        case_id: str,
        system: str,
        final_result: FinalRCAResult,
        ground_truth: GroundTruth,
    ) -> CaseEvaluationResult:
        if final_result.case_id != case_id:
            raise ValueError(
                f"case_id mismatch: FinalRCAResult={final_result.case_id!r} "
                f"expected={case_id!r}"
            )
        if ground_truth.case_id != case_id:
            raise ValueError(
                f"case_id mismatch: GroundTruth={ground_truth.case_id!r} "
                f"expected={case_id!r}"
            )

        notes: List[str] = []
        target_service = ground_truth.root_cause_service

        predicted_rank: Optional[int] = None
        for idx, grounded in enumerate(final_result.candidates, start=1):
            if grounded.candidate.service == target_service:
                predicted_rank = idx
                break

        if predicted_rank is None:
            notes.append(
                f"root_cause_service {target_service!r} not present among "
                f"{len(final_result.candidates)} candidates for case {case_id!r}."
            )

        top_1 = predicted_rank == 1
        top_3 = predicted_rank is not None and predicted_rank <= 3
        top_5 = predicted_rank is not None and predicted_rank <= 5

        supported = 0
        unsupported = 0
        uncertain = 0
        for grounded in final_result.candidates:
            if grounded.candidate.service != target_service:
                continue
            supported += len(grounded.supporting_claim_ids)
            unsupported += len(grounded.contradicting_claim_ids)
            uncertain += len(grounded.uncertain_claim_ids)

        definitive = supported + unsupported
        total_claims = supported + unsupported + uncertain

        grounding_precision = (supported / definitive) if definitive > 0 else None
        grounding_recall = (supported / total_claims) if total_claims > 0 else None

        if definitive == 0:
            notes.append(
                f"No definitive (supported/unsupported) claims about "
                f"{target_service!r} for case {case_id!r}; grounding_precision is undefined."
            )
        if total_claims == 0:
            notes.append(
                f"No claims at all about {target_service!r} for case "
                f"{case_id!r}; grounding_recall is undefined."
            )

        return CaseEvaluationResult(
            case_id=case_id,
            system=system,
            fault_type=ground_truth.fault_type,
            root_cause_service=target_service,
            predicted_rank=predicted_rank,
            top_1_accuracy=top_1,
            top_3_accuracy=top_3,
            top_5_accuracy=top_5,
            grounding_precision=grounding_precision,
            grounding_recall=grounding_recall,
            notes=notes,
        )

    def summarize(
        self, per_case: List[CaseEvaluationResult]
    ) -> Tuple[OverallEvaluationSummary, Dict[str, SystemEvaluationSummary]]:
        overall_agg = self._aggregate(per_case)
        overall = OverallEvaluationSummary(**overall_agg)

        by_system_cases: Dict[str, List[CaseEvaluationResult]] = defaultdict(list)
        for case in per_case:
            by_system_cases[case.system].append(case)

        by_system: Dict[str, SystemEvaluationSummary] = {}
        for system, cases in by_system_cases.items():
            agg = self._aggregate(cases)
            by_system[system] = SystemEvaluationSummary(system=system, **agg)

        return overall, by_system

    @staticmethod
    def _aggregate(cases: List[CaseEvaluationResult]) -> dict:
        n = len(cases)
        if n == 0:
            return {
                "case_count": 0,
                "top_1_accuracy": 0.0,
                "top_3_accuracy": 0.0,
                "top_5_accuracy": 0.0,
                "mean_grounding_precision": None,
                "mean_grounding_recall": None,
            }

        top_1 = sum(1 for c in cases if c.top_1_accuracy) / n
        top_3 = sum(1 for c in cases if c.top_3_accuracy) / n
        top_5 = sum(1 for c in cases if c.top_5_accuracy) / n

        precisions = [c.grounding_precision for c in cases if c.grounding_precision is not None]
        recalls = [c.grounding_recall for c in cases if c.grounding_recall is not None]

        return {
            "case_count": n,
            "top_1_accuracy": top_1,
            "top_3_accuracy": top_3,
            "top_5_accuracy": top_5,
            "mean_grounding_precision": (sum(precisions) / len(precisions)) if precisions else None,
            "mean_grounding_recall": (sum(recalls) / len(recalls)) if recalls else None,
        }