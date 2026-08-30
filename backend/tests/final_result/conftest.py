from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.schemas.claims import ClaimVerdict, ClaimVerificationResult
from app.schemas.reasoning import CandidateCause, RCACandidateResult

AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def candidate_result() -> RCACandidateResult:
    return RCACandidateResult(
        case_id="case-0001",
        as_of=AS_OF,
        method="deterministic",
        candidates=[
            CandidateCause(
                service="cartservice",
                score=1.4,
                supporting_evidence=["cartservice_cpu", "cartservice_latency"],
                graph_support=["frontend"],
            ),
            CandidateCause(
                service="frontend",
                score=0.2,
                supporting_evidence=["frontend_cpu"],
                graph_support=["cartservice"],
            ),
            CandidateCause(
                service="checkoutservice",
                score=0.1,
                supporting_evidence=["checkoutservice_cpu"],
                graph_support=None,
            ),
        ],
        notes=["deterministic candidate generation note"],
    )


def make_verification_result(verdicts, case_id="case-0001", as_of=AS_OF, notes=None):
    return ClaimVerificationResult(
        case_id=case_id,
        as_of=as_of,
        verdicts=verdicts,
        notes=notes or ["claim verification note"],
    )


@pytest.fixture
def verification_result_factory():
    return make_verification_result


@pytest.fixture
def verdict_factory():
    def _make(claim_id: str, candidate_service: str, status: str) -> ClaimVerdict:
        matched = ["some_column"] if status == "supported" else []
        unmatched = ["some_column"] if status == "unsupported" else []
        return ClaimVerdict(
            claim_id=claim_id,
            candidate_service=candidate_service,
            status=status,
            matched_evidence=matched,
            unmatched_evidence=unmatched,
        )

    return _make
