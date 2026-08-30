from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pytest

from app.schemas.final_result import CandidateGroundingStatus, FinalRCAResult, GroundedCandidate
from app.schemas.ground_truth import GroundTruth
from app.schemas.reasoning import CandidateCause

AS_OF = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _grounded(
    service: str,
    status: CandidateGroundingStatus,
    supporting: List[str] = (),
    contradicting: List[str] = (),
    uncertain: List[str] = (),
) -> GroundedCandidate:
    return GroundedCandidate(
        candidate=CandidateCause(service=service, score=1.0, supporting_evidence=[]),
        grounding_status=status,
        supporting_claim_ids=list(supporting),
        contradicting_claim_ids=list(contradicting),
        uncertain_claim_ids=list(uncertain),
    )


@pytest.fixture
def ground_truth() -> GroundTruth:
    return GroundTruth(case_id="case-0001", root_cause_service="cartservice", fault_type="cpu")


@pytest.fixture
def final_result_top1_supported() -> FinalRCAResult:
    """cartservice ranked #1: 2 supported claims, 1 contradicting, 0 uncertain."""
    return FinalRCAResult(
        case_id="case-0001",
        as_of=AS_OF,
        candidates=[
            _grounded("cartservice", "uncertain", supporting=["c1", "c2"], contradicting=["c3"]),
            _grounded("checkoutservice", "uncertain"),
        ],
        notes=[],
    )


@pytest.fixture
def final_result_rank3() -> FinalRCAResult:
    return FinalRCAResult(
        case_id="case-0001",
        as_of=AS_OF,
        candidates=[
            _grounded("checkoutservice", "uncertain"),
            _grounded("paymentservice", "uncertain"),
            _grounded("cartservice", "supported", supporting=["c1"]),
        ],
        notes=[],
    )


@pytest.fixture
def final_result_not_in_candidates() -> FinalRCAResult:
    """root_cause_service (cartservice) does not appear among candidates."""
    return FinalRCAResult(
        case_id="case-0001",
        as_of=AS_OF,
        candidates=[
            _grounded("checkoutservice", "uncertain"),
            _grounded("paymentservice", "uncertain"),
        ],
        notes=[],
    )