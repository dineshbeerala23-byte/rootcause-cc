# ============================================================
# app/services/evaluation/service.py  (NEW)
# ============================================================
"""
Evaluation orchestration layer.

This is the ONLY new evaluation-milestone module permitted to call
adapter.get_ground_truth() -- EvaluationScorer never touches the adapter
or ground truth directly. Same "orchestrator fetches, scorer combines"
split as FinalResultAssemblyService / ClaimVerificationService.

Read-only with respect to RCA results: this never re-ranks or modifies
a FinalRCAResult, it only runs the existing pipeline per case and scores
what comes back.
"""

from __future__ import annotations

from typing import List, Optional

from app.core.exceptions import RootCauseError
from app.schemas.evaluation import EvaluationRunResult
from app.schemas.reasoning import ReasoningMethod
from app.services.datasets.base import DatasetAdapter
from app.services.evaluation.scorer import EvaluationScorer
from app.services.pipeline.service import RCAPipelineService


class EvaluationService:
    def __init__(
        self,
        adapter: DatasetAdapter,
        pipeline_service: RCAPipelineService,
        scorer: Optional[EvaluationScorer] = None,
    ) -> None:
        self._adapter = adapter
        self._pipeline_service = pipeline_service
        self._scorer = scorer or EvaluationScorer()

    async def evaluate_cases(
        self,
        case_ids: List[str],
        mode: ReasoningMethod = "deterministic",
    ) -> EvaluationRunResult:
        cases_by_id = {c.case_id: c for c in self._adapter.list_cases()}

        per_case = []
        notes: List[str] = []

        for case_id in case_ids:
            metadata = cases_by_id.get(case_id)
            if metadata is None:
                notes.append(f"case_id {case_id!r} not found via list_cases(); skipped.")
                continue

            try:
                ground_truth = self._adapter.get_ground_truth(case_id)
            except RootCauseError as exc:
                notes.append(f"case_id {case_id!r}: failed to fetch ground truth ({exc.message}); skipped.")
                continue

            try:
                analysis_response = await self._pipeline_service.analyze(case_id, mode=mode)
            except RootCauseError as exc:
                notes.append(f"case_id {case_id!r}: pipeline failed ({exc.message}); skipped.")
                continue

            case_result = self._scorer.score_case(
                case_id=metadata.case_id,
                system=metadata.system,
                final_result=analysis_response.result,
                ground_truth=ground_truth,
            )
            per_case.append(case_result)

        overall, by_system = self._scorer.summarize(per_case)

        return EvaluationRunResult(
            per_case=per_case,
            overall=overall,
            by_system=by_system,
            notes=notes,
        )