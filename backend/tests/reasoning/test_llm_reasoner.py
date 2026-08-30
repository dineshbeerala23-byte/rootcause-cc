import json

import pytest

from app.services.llm.base import LLMProvider
from app.services.reasoning.deterministic import DeterministicCandidateGenerator
from app.services.reasoning.llm_reasoner import LLMRationaleAnnotator


class FakeLLMProvider(LLMProvider):
    def __init__(self, response: str = "", raise_error: bool = False):
        self._response = response
        self._raise_error = raise_error

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self._raise_error:
            raise RuntimeError("simulated provider failure")
        return self._response


def _deterministic_result(incident, evidence, graph):
    return DeterministicCandidateGenerator().generate(incident, evidence, graph)


@pytest.mark.asyncio
async def test_valid_rationale_is_attached_without_changing_order_or_scores(
    incident, evidence, graph_with_edges
):
    deterministic = _deterministic_result(incident, evidence, graph_with_edges)
    original_services = [c.service for c in deterministic.candidates]
    original_scores = [c.score for c in deterministic.candidates]

    response = json.dumps(
        {
            "rationales": {
                "cartservice": "cartservice shows the largest deviation in CPU and memory.",
                "redis": "redis is a direct dependency of cartservice.",
            }
        }
    )
    annotator = LLMRationaleAnnotator(FakeLLMProvider(response=response))

    result = await annotator.annotate(deterministic, evidence, graph_with_edges)

    assert result.method == "llm-assisted"
    assert [c.service for c in result.candidates] == original_services
    assert [c.score for c in result.candidates] == original_scores
    assert result.candidates[0].rationale == (
        "cartservice shows the largest deviation in CPU and memory."
    )
    assert result.candidates[1].rationale == "redis is a direct dependency of cartservice."


@pytest.mark.asyncio
async def test_malformed_json_falls_back_to_deterministic_with_note(
    incident, evidence, graph_with_edges
):
    deterministic = _deterministic_result(incident, evidence, graph_with_edges)
    annotator = LLMRationaleAnnotator(FakeLLMProvider(response="not json at all"))

    result = await annotator.annotate(deterministic, evidence, graph_with_edges)

    assert result.method == "deterministic"
    assert result.candidates == deterministic.candidates
    assert any("invalid" in note.lower() for note in result.notes)


@pytest.mark.asyncio
async def test_unknown_service_in_rationale_falls_back_with_note(
    incident, evidence, graph_with_edges
):
    deterministic = _deterministic_result(incident, evidence, graph_with_edges)
    response = json.dumps({"rationales": {"some-service-not-in-candidates": "hallucinated"}})
    annotator = LLMRationaleAnnotator(FakeLLMProvider(response=response))

    result = await annotator.annotate(deterministic, evidence, graph_with_edges)

    assert result.method == "deterministic"
    assert result.candidates == deterministic.candidates
    assert any("unknown service" in note.lower() for note in result.notes)


@pytest.mark.asyncio
async def test_provider_failure_falls_back_to_deterministic_with_note(
    incident, evidence, graph_with_edges
):
    deterministic = _deterministic_result(incident, evidence, graph_with_edges)
    annotator = LLMRationaleAnnotator(FakeLLMProvider(raise_error=True))

    result = await annotator.annotate(deterministic, evidence, graph_with_edges)

    assert result.method == "deterministic"
    assert result.candidates == deterministic.candidates
    assert any("failed" in note.lower() for note in result.notes)


@pytest.mark.asyncio
async def test_empty_candidate_set_short_circuits_without_calling_provider(
    incident, evidence, graph_without_edges
):
    from app.schemas.reasoning import RCACandidateResult

    empty_result = RCACandidateResult(
        case_id=incident.case_id, as_of=evidence.as_of, method="deterministic", candidates=[], notes=[]
    )

    class ExplodingProvider(LLMProvider):
        async def complete(self, system_prompt: str, user_prompt: str) -> str:
            raise AssertionError("should never be called for an empty candidate set")

    annotator = LLMRationaleAnnotator(ExplodingProvider())
    result = await annotator.annotate(empty_result, evidence, graph_without_edges)

    assert result is empty_result