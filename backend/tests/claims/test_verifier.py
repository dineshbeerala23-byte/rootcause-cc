from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.claims import Claim, ClaimGenerationResult
from app.services.claims.verifier import DeterministicClaimVerifier

AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _generation_result(claims):
    return ClaimGenerationResult(
        case_id="case-0001",
        as_of=AS_OF,
        method="llm-generated",
        claims=claims,
        notes=[],
    )


def test_all_cited_evidence_found_and_attributed_is_supported(evidence_bundle):
    claim = Claim(
        claim_id="c1",
        candidate_service="cartservice",
        statement="cartservice shows elevated CPU and latency.",
        cited_evidence=["cartservice_cpu", "cartservice_latency"],
    )
    result = DeterministicClaimVerifier().verify(_generation_result([claim]), evidence_bundle)

    assert len(result.verdicts) == 1
    verdict = result.verdicts[0]
    assert verdict.status == "supported"
    assert verdict.matched_evidence == ["cartservice_cpu", "cartservice_latency"]
    assert verdict.unmatched_evidence == []


def test_no_cited_evidence_found_is_unsupported(evidence_bundle):
    claim = Claim(
        claim_id="c1",
        candidate_service="cartservice",
        statement="cartservice is implicated by evidence not actually present.",
        cited_evidence=["nonexistent_column"],
    )
    result = DeterministicClaimVerifier().verify(_generation_result([claim]), evidence_bundle)

    verdict = result.verdicts[0]
    assert verdict.status == "unsupported"
    assert verdict.matched_evidence == []
    assert verdict.unmatched_evidence == ["nonexistent_column"]


def test_partial_match_is_uncertain(evidence_bundle):
    claim = Claim(
        claim_id="c1",
        candidate_service="cartservice",
        statement="cartservice shows elevated CPU and an unrelated signal.",
        cited_evidence=["cartservice_cpu", "nonexistent_column"],
    )
    result = DeterministicClaimVerifier().verify(_generation_result([claim]), evidence_bundle)

    verdict = result.verdicts[0]
    assert verdict.status == "uncertain"
    assert verdict.matched_evidence == ["cartservice_cpu"]
    assert verdict.unmatched_evidence == ["nonexistent_column"]


def test_empty_cited_evidence_is_uncertain(evidence_bundle):
    claim = Claim(
        claim_id="c1",
        candidate_service="cartservice",
        statement="cartservice is likely responsible.",
        cited_evidence=[],
    )
    result = DeterministicClaimVerifier().verify(_generation_result([claim]), evidence_bundle)

    verdict = result.verdicts[0]
    assert verdict.status == "uncertain"
    assert verdict.matched_evidence == []
    assert verdict.unmatched_evidence == []


def test_evidence_found_but_attributed_to_different_service_is_not_supported(evidence_bundle):
    # frontend_cpu exists in the bundle, but is attributed to "frontend", not "cartservice".
    claim = Claim(
        claim_id="c1",
        candidate_service="cartservice",
        statement="cartservice is implicated by another service's CPU metric.",
        cited_evidence=["frontend_cpu"],
    )
    result = DeterministicClaimVerifier().verify(_generation_result([claim]), evidence_bundle)

    verdict = result.verdicts[0]
    assert verdict.status == "unsupported"
    assert verdict.unmatched_evidence == ["frontend_cpu"]


def test_unattributed_evidence_column_never_supports_a_claim(evidence_bundle):
    # unattributed_cpu exists but has service=None -- can never match any candidate_service.
    claim = Claim(
        claim_id="c1",
        candidate_service="cartservice",
        statement="cartservice is implicated by an unattributed signal.",
        cited_evidence=["unattributed_cpu"],
    )
    result = DeterministicClaimVerifier().verify(_generation_result([claim]), evidence_bundle)

    verdict = result.verdicts[0]
    assert verdict.status == "unsupported"


def test_no_claims_produces_empty_verdicts_with_note(evidence_bundle):
    result = DeterministicClaimVerifier().verify(_generation_result([]), evidence_bundle)

    assert result.verdicts == []
    assert any("No claims to verify" in note for note in result.notes)


def test_as_of_is_preserved_verbatim_from_evidence_bundle(evidence_bundle):
    claim = Claim(
        claim_id="c1",
        candidate_service="cartservice",
        statement="cartservice shows elevated CPU.",
        cited_evidence=["cartservice_cpu"],
    )
    result = DeterministicClaimVerifier().verify(_generation_result([claim]), evidence_bundle)

    assert result.as_of == evidence_bundle.as_of


def test_generation_notes_are_carried_through(evidence_bundle):
    generation_result = ClaimGenerationResult(
        case_id="case-0001",
        as_of=AS_OF,
        method="llm-generated",
        claims=[],
        notes=["a prior note from generation"],
    )
    result = DeterministicClaimVerifier().verify(generation_result, evidence_bundle)

    assert "a prior note from generation" in result.notes
