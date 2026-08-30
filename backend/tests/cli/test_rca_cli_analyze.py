from __future__ import annotations

"""
CLI test for `python -m app.cli.rcaeval_cli analyze <case_id> [--mode ...]`.

NOTE: I do not have visibility into this repository's existing CLI test
conventions (if any exist for list/inspect/process) or the real
constructor signature of app.core.exceptions.RootCauseError -- the error
test below guesses `RootCauseError(message=...)`; if the real signature
differs, only that one test needs adjusting, not the approach.
"""

from datetime import datetime, timezone

import pytest

from app.cli import rcaeval_cli
from app.core.exceptions import RootCauseError
from app.schemas.claims import ClaimVerdict, ClaimVerificationResult
from app.schemas.final_result import FinalRCAResult, GroundedCandidate
from app.schemas.pipeline import RCAAnalysisResponse
from app.schemas.reasoning import CandidateCause

AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakePipelineService:
    def __init__(self, raise_error: bool = False):
        self._raise_error = raise_error
        self.last_case_id = None
        self.last_mode = None

    async def analyze(self, case_id: str, mode: str = "deterministic") -> RCAAnalysisResponse:
        self.last_case_id = case_id
        self.last_mode = mode
        if self._raise_error:
            raise RootCauseError(message=f"No such case: {case_id}")

        candidate = CandidateCause(
            service="cartservice",
            score=0.8,
            supporting_evidence=["cartservice_cpu"],
            graph_support=None,
        )
        grounded = GroundedCandidate(
            candidate=candidate,
            grounding_status="supported",
            supporting_claim_ids=["c1"],
            contradicting_claim_ids=[],
            uncertain_claim_ids=[],
        )
        result = FinalRCAResult(case_id=case_id, as_of=AS_OF, candidates=[grounded], notes=[])
        return RCAAnalysisResponse(
            case_id=case_id,
            mode=mode,
            provider="fake-provider",
            model="fake-model",
            result=result,
        )


def test_analyze_command_success(monkeypatch, capsys):
    fake_service = FakePipelineService()
    monkeypatch.setattr(rcaeval_cli, "_build_pipeline_service", lambda: fake_service)

    exit_code = rcaeval_cli.main(["analyze", "case-0001"])

    assert exit_code == 0
    assert fake_service.last_case_id == "case-0001"
    assert fake_service.last_mode == "deterministic"

    captured = capsys.readouterr()
    assert '"case_id": "case-0001"' in captured.out
    assert '"grounding_status": "supported"' in captured.out


def test_analyze_command_mode_flag_is_passed_through(monkeypatch, capsys):
    fake_service = FakePipelineService()
    monkeypatch.setattr(rcaeval_cli, "_build_pipeline_service", lambda: fake_service)

    exit_code = rcaeval_cli.main(["analyze", "case-0001", "--mode", "llm-assisted"])

    assert exit_code == 0
    assert fake_service.last_mode == "llm-assisted"


def test_analyze_command_unknown_case_id_produces_error_exit_code(monkeypatch, capsys):
    fake_service = FakePipelineService(raise_error=True)
    monkeypatch.setattr(rcaeval_cli, "_build_pipeline_service", lambda: fake_service)

    exit_code = rcaeval_cli.main(["analyze", "case-9999"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error" in captured.err
