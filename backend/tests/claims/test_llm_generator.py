from __future__ import annotations

import pytest

from app.schemas.reasoning import RCACandidateResult
from app.services.claims.llm_generator import LLMClaimGenerator


@pytest.mark.asyncio
async def test_valid_claims_are_parsed(
    candidate_result, evidence_bundle, service_graph, llm_provider_factory, claims_payload_factory
):
    payload = claims_payload_factory(
        [
            {
                "claim_id": "c1",
                "candidate_service": "cartservice",
                "statement": "cartservice shows elevated CPU and latency.",
                "cited_evidence": ["cartservice_cpu", "cartservice_latency"],
            }
        ]
    )
    provider = llm_provider_factory(response=payload)
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(candidate_result, evidence_bundle, service_graph)

    assert result.method == "llm-generated"
    assert len(result.claims) == 1
    assert result.claims[0].claim_id == "c1"
    assert result.claims[0].candidate_service == "cartservice"
    assert result.claims[0].cited_evidence == ["cartservice_cpu", "cartservice_latency"]
    assert result.as_of == evidence_bundle.as_of


@pytest.mark.asyncio
async def test_claim_citing_evidence_outside_candidate_supporting_evidence_is_dropped(
    candidate_result, evidence_bundle, service_graph, llm_provider_factory, claims_payload_factory
):
    payload = claims_payload_factory(
        [
            {
                "claim_id": "c1",
                "candidate_service": "cartservice",
                "statement": "cartservice is implicated by frontend's metric.",
                # frontend_cpu is NOT in cartservice's supporting_evidence.
                "cited_evidence": ["frontend_cpu"],
            },
            {
                "claim_id": "c2",
                "candidate_service": "cartservice",
                "statement": "cartservice shows elevated CPU.",
                "cited_evidence": ["cartservice_cpu"],
            },
        ]
    )
    provider = llm_provider_factory(response=payload)
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(candidate_result, evidence_bundle, service_graph)

    assert len(result.claims) == 1
    assert result.claims[0].claim_id == "c2"
    assert any("c1" in note for note in result.notes)


@pytest.mark.asyncio
async def test_claim_referencing_unknown_candidate_service_is_dropped(
    candidate_result, evidence_bundle, service_graph, llm_provider_factory, claims_payload_factory
):
    payload = claims_payload_factory(
        [
            {
                "claim_id": "c1",
                "candidate_service": "some-service-not-in-candidates",
                "statement": "an invented service is implicated.",
                "cited_evidence": [],
            }
        ]
    )
    provider = llm_provider_factory(response=payload)
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(candidate_result, evidence_bundle, service_graph)

    assert result.claims == []
    assert any("unknown candidate service" in note for note in result.notes)


@pytest.mark.asyncio
async def test_empty_response_fails_safely(
    candidate_result, evidence_bundle, service_graph, llm_provider_factory
):
    provider = llm_provider_factory(response="")
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(candidate_result, evidence_bundle, service_graph)

    assert result.claims == []
    assert any("empty response" in note for note in result.notes)


@pytest.mark.asyncio
async def test_unparseable_json_fails_safely(
    candidate_result, evidence_bundle, service_graph, llm_provider_factory
):
    provider = llm_provider_factory(response="not valid json{{{")
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(candidate_result, evidence_bundle, service_graph)

    assert result.claims == []
    assert any("could not parse JSON" in note for note in result.notes)


@pytest.mark.asyncio
async def test_provider_exception_fails_safely_and_never_raises(
    candidate_result, evidence_bundle, service_graph, llm_provider_factory
):
    provider = llm_provider_factory(error=RuntimeError("provider unavailable"))
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(candidate_result, evidence_bundle, service_graph)

    assert result.claims == []
    assert any("LLM claim generation failed" in note for note in result.notes)


@pytest.mark.asyncio
async def test_no_candidates_produces_empty_result_without_calling_provider(
    evidence_bundle, service_graph, llm_provider_factory, claims_payload_factory
):
    empty_candidate_result = RCACandidateResult(
        case_id="case-0001",
        as_of=evidence_bundle.as_of,
        method="deterministic",
        candidates=[],
        notes=[],
    )
    provider = llm_provider_factory(response=claims_payload_factory([]))
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(empty_candidate_result, evidence_bundle, service_graph)

    assert result.claims == []
    assert provider.last_system_prompt is None  # provider was never called


@pytest.mark.asyncio
async def test_duplicate_claim_ids_second_is_dropped(
    candidate_result, evidence_bundle, service_graph, llm_provider_factory, claims_payload_factory
):
    payload = claims_payload_factory(
        [
            {
                "claim_id": "c1",
                "candidate_service": "cartservice",
                "statement": "cartservice shows elevated CPU.",
                "cited_evidence": ["cartservice_cpu"],
            },
            {
                "claim_id": "c1",
                "candidate_service": "cartservice",
                "statement": "duplicate claim id.",
                "cited_evidence": ["cartservice_latency"],
            },
        ]
    )
    provider = llm_provider_factory(response=payload)
    generator = LLMClaimGenerator(provider)

    result = await generator.generate(candidate_result, evidence_bundle, service_graph)

    assert len(result.claims) == 1
    assert any("duplicate claim_id" in note for note in result.notes)
