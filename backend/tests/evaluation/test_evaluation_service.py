from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict

import pytest

from app.core.exceptions import DatasetCaseNotFoundError
from app.schemas.final_result import FinalRCAResult, GroundedCandidate
from app.schemas.ground_truth import GroundTruth
from app.schemas.reasoning import CandidateCause
from app.services.evaluation.service import EvaluationService
from app.schemas.pipeline import RCAAnalysisResponse

AS_OF = datetime(2024, 1, 1, tzinfo=timezone.utc)


class _FakeAdapter:
    """
    Duck-typed test double implementing only what EvaluationService uses
    (list_cases, get_ground_truth). Deliberately not a real CaseMetadata/
    DatasetAdapter instance -- this milestone doesn't need
    ModalityAvailability or any other unrelated adapter surface.
    """

    def __init__(self, cases: Dict[str, SimpleNamespace], ground_truths: Dict[str, GroundTruth]):
        self._cases = cases
        self._ground_truths = ground_truths

    def list_cases(self):
        return list(self._cases.values())

    def get_ground_truth(self, case_id: str) -> GroundTruth:
        if case_id not in self._ground_truths:
            raise DatasetCaseNotFoundError(case_id)
        return self._ground_truths[case_id]


class _FakePipeline:
    """
    Test double for RCAPipelineService. Must match the real
    analyze() return type -- RCAAnalysisResponse (an envelope around
    FinalRCAResult), not a bare FinalRCAResult -- since
    EvaluationService.evaluate_cases() unwraps `.result` from whatever
    this returns.
    """

    def __init__(self, results: Dict[str, FinalRCAResult]):
        self._results = results

    async def analyze(self, case_id: str, mode: str = "deterministic") -> RCAAnalysisResponse:
        if case_id not in self._results:
            raise DatasetCaseNotFoundError(case_id)
        final_result = self._results[case_id]
        return RCAAnalysisResponse(
            case_id=case_id,
            mode=mode,
            provider="fake",
            model="fake-model",
            result=final_result,
        )

def _case(case_id: str, system: str) -> SimpleNamespace:
    return SimpleNamespace(case_id=case_id, system=system)


def _final_result(case_id: str, service: str) -> FinalRCAResult:
    return FinalRCAResult(
        case_id=case_id,
        as_of=AS_OF,
        candidates=[
            GroundedCandidate(
                candidate=CandidateCause(service=service, score=1.0, supporting_evidence=[]),
                grounding_status="supported",
                supporting_claim_ids=["c1"],
            )
        ],
        notes=[],
    )


@pytest.mark.asyncio
async def test_evaluate_cases_happy_path():
    cases = {"case-0001": _case("case-0001", "online-boutique")}
    ground_truths = {
        "case-0001": GroundTruth(case_id="case-0001", root_cause_service="cartservice", fault_type="cpu")
    }
    results = {"case-0001": _final_result("case-0001", "cartservice")}

    service = EvaluationService(
        adapter=_FakeAdapter(cases, ground_truths),
        pipeline_service=_FakePipeline(results),
    )

    run_result = await service.evaluate_cases(["case-0001"])

    assert len(run_result.per_case) == 1
    assert run_result.per_case[0].top_1_accuracy is True
    assert run_result.overall.case_count == 1
    assert "online-boutique" in run_result.by_system
    assert run_result.notes == []


@pytest.mark.asyncio
async def test_evaluate_cases_skips_unknown_case_id():
    service = EvaluationService(adapter=_FakeAdapter({}, {}), pipeline_service=_FakePipeline({}))

    run_result = await service.evaluate_cases(["case-9999"])

    assert run_result.per_case == []
    assert run_result.overall.case_count == 0
    assert any("not found" in n for n in run_result.notes)


@pytest.mark.asyncio
async def test_evaluate_cases_skips_case_when_ground_truth_missing():
    cases = {"case-0001": _case("case-0001", "online-boutique")}
    service = EvaluationService(adapter=_FakeAdapter(cases, {}), pipeline_service=_FakePipeline({}))

    run_result = await service.evaluate_cases(["case-0001"])

    assert run_result.per_case == []
    assert any("ground truth" in n for n in run_result.notes)


@pytest.mark.asyncio
async def test_evaluate_cases_skips_case_when_pipeline_fails():
    cases = {"case-0001": _case("case-0001", "online-boutique")}
    ground_truths = {
        "case-0001": GroundTruth(case_id="case-0001", root_cause_service="cartservice", fault_type="cpu")
    }
    service = EvaluationService(adapter=_FakeAdapter(cases, ground_truths), pipeline_service=_FakePipeline({}))

    run_result = await service.evaluate_cases(["case-0001"])

    assert run_result.per_case == []
    assert any("pipeline failed" in n for n in run_result.notes)