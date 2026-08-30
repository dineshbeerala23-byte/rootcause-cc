from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

import pytest

from app.schemas.evidence import EvidenceBundle, EvidenceItem
from app.schemas.graph import ServiceGraph
from app.schemas.reasoning import CandidateCause, RCACandidateResult
from app.services.llm.base import LLMProvider

AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeLLMProvider(LLMProvider):
    """
    Test double for LLMProvider. Returns a pre-set response, or raises a
    pre-set exception, so tests never make a real network call.
    """

    def __init__(self, response: Optional[str] = None, error: Optional[Exception] = None):
        self._response = response
        self._error = error
        self.last_system_prompt: Optional[str] = None
        self.last_user_prompt: Optional[str] = None

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        if self._error is not None:
            raise self._error
        return self._response or ""


def make_claims_payload(claims: List[dict]) -> str:
    return json.dumps({"claims": claims})


@pytest.fixture
def llm_provider_factory():
    """
    Returns a factory for FakeLLMProvider so tests don't need to import
    the class directly across a module boundary.
    """

    def _make(response: Optional[str] = None, error: Optional[Exception] = None) -> FakeLLMProvider:
        return FakeLLMProvider(response=response, error=error)

    return _make


@pytest.fixture
def claims_payload_factory():
    """Returns make_claims_payload without requiring a cross-module import."""
    return make_claims_payload


@pytest.fixture
def evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        case_id="case-0001",
        as_of=AS_OF,
        modalities_used=["metrics"],
        items=[
            EvidenceItem(
                service="cartservice",
                metric_type="cpu",
                column_name="cartservice_cpu",
                summary={"mean": 0.9},
                score=0.8,
            ),
            EvidenceItem(
                service="cartservice",
                metric_type="latency",
                column_name="cartservice_latency",
                summary={"mean": 120.0},
                score=0.6,
            ),
            EvidenceItem(
                service="frontend",
                metric_type="cpu",
                column_name="frontend_cpu",
                summary={"mean": 0.3},
                score=0.2,
            ),
            EvidenceItem(
                service=None,
                metric_type="cpu",
                column_name="unattributed_cpu",
                summary={"mean": 0.1},
                score=0.05,
            ),
        ],
    )


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
        ],
        notes=[],
    )


@pytest.fixture
def service_graph() -> ServiceGraph:
    return ServiceGraph(
        system="online-boutique",
        nodes=[],
        edges=[],
        notes=["edges not yet researched for this fixture"],
    )
