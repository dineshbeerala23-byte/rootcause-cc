from __future__ import annotations

"""
API test for POST /api/v1/rca/{case_id}/analyze.

NOTE: this file uses a standard FastAPI TestClient + dependency-override
pattern. I do not have visibility into this repository's actual
app/main.py, existing tests/api/ fixtures, or how extraction's endpoint
is tested -- so the `app` import path and any existing `client` fixture
name below are best-effort guesses and may need adjusting to match real
conventions (e.g. an existing conftest.py may already provide a `client`
fixture, in which case the local one here should be removed in favor of
it to avoid duplication).
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_pipeline_service
from app.main import app  # best-effort import path -- verify against real app entrypoint
from app.schemas.claims import ClaimVerdict, ClaimVerificationResult
from app.schemas.final_result import FinalRCAResult, GroundedCandidate
from app.schemas.pipeline import RCAAnalysisResponse
from app.schemas.reasoning import CandidateCause

AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakePipelineService:
    async def analyze(self, case_id: str, mode: str = "deterministic") -> RCAAnalysisResponse:
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


@pytest.fixture
def client():
    app.dependency_overrides[get_pipeline_service] = lambda: FakePipelineService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_pipeline_service, None)


def test_analyze_endpoint_returns_expected_shape(client):
    response = client.post("/api/v1/rca/case-0001/analyze", json={"mode": "deterministic"})

    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "case-0001"
    assert body["mode"] == "deterministic"
    assert body["result"]["candidates"][0]["grounding_status"] == "supported"


def test_analyze_endpoint_defaults_mode_to_deterministic_with_empty_body(client):
    response = client.post("/api/v1/rca/case-0001/analyze", json={})

    assert response.status_code == 200
    assert response.json()["mode"] == "deterministic"


def test_analyze_endpoint_response_contains_no_ground_truth_fields(client):
    response = client.post("/api/v1/rca/case-0001/analyze", json={})

    body_text = response.text
    assert "root_cause_service" not in body_text
    assert "fault_type" not in body_text
    assert "root_cause_indicator" not in body_text
