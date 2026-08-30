from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone

from app.services.final_result.service import FinalResultAssemblyService


def test_all_supported_claims_yield_supported_status(
    candidate_result, verification_result_factory, verdict_factory
):
    verification = verification_result_factory(
        [
            verdict_factory("c1", "cartservice", "supported"),
            verdict_factory("c2", "cartservice", "supported"),
        ]
    )
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    cart = next(g for g in result.candidates if g.candidate.service == "cartservice")
    assert cart.grounding_status == "supported"
    assert cart.supporting_claim_ids == ["c1", "c2"]
    assert cart.contradicting_claim_ids == []
    assert cart.uncertain_claim_ids == []


def test_any_unsupported_claim_yields_unsupported_status(
    candidate_result, verification_result_factory, verdict_factory
):
    verification = verification_result_factory(
        [
            verdict_factory("c1", "cartservice", "supported"),
            verdict_factory("c2", "cartservice", "unsupported"),
        ]
    )
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    cart = next(g for g in result.candidates if g.candidate.service == "cartservice")
    assert cart.grounding_status == "unsupported"
    assert cart.contradicting_claim_ids == ["c2"]


def test_mixed_supported_and_uncertain_without_unsupported_yields_uncertain(
    candidate_result, verification_result_factory, verdict_factory
):
    verification = verification_result_factory(
        [
            verdict_factory("c1", "cartservice", "supported"),
            verdict_factory("c2", "cartservice", "uncertain"),
        ]
    )
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    cart = next(g for g in result.candidates if g.candidate.service == "cartservice")
    assert cart.grounding_status == "uncertain"


def test_zero_claims_for_candidate_yields_uncertain(
    candidate_result, verification_result_factory, verdict_factory
):
    # Only cartservice has verdicts; frontend and checkoutservice have none.
    verification = verification_result_factory(
        [verdict_factory("c1", "cartservice", "supported")]
    )
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    frontend = next(g for g in result.candidates if g.candidate.service == "frontend")
    checkout = next(g for g in result.candidates if g.candidate.service == "checkoutservice")
    assert frontend.grounding_status == "uncertain"
    assert frontend.supporting_claim_ids == []
    assert checkout.grounding_status == "uncertain"


def test_candidate_order_is_preserved_exactly(
    candidate_result, verification_result_factory, verdict_factory
):
    # Give checkoutservice a "supported" verdict -- it should NOT move to
    # the front despite outranking frontend's "uncertain" status.
    verification = verification_result_factory(
        [
            verdict_factory("c1", "cartservice", "unsupported"),
            verdict_factory("c2", "checkoutservice", "supported"),
        ]
    )
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    services_in_order = [g.candidate.service for g in result.candidates]
    assert services_in_order == ["cartservice", "frontend", "checkoutservice"]


def test_as_of_is_taken_from_verification_result(candidate_result, verification_result_factory):
    verification = verification_result_factory([])
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    assert result.as_of == verification.as_of


def test_notes_are_cumulative_across_reasoning_and_verification_and_assembly(
    candidate_result, verification_result_factory
):
    verification = verification_result_factory([], notes=["verification-specific note"])
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    assert "deterministic candidate generation note" in result.notes  # from candidate_result
    assert "verification-specific note" in result.notes  # from verification_result


def test_case_id_mismatch_produces_note_not_silent_resolution(
    candidate_result, verification_result_factory
):
    mismatched = verification_result_factory([], case_id="case-9999")
    result = FinalResultAssemblyService().assemble(candidate_result, mismatched)

    assert result.case_id == candidate_result.case_id  # RCACandidateResult wins, per design
    assert any("case_id mismatch" in note for note in result.notes)


def test_as_of_mismatch_produces_note_not_silent_resolution(
    candidate_result, verification_result_factory
):
    different_as_of = datetime(2026, 6, 1, tzinfo=timezone.utc)
    mismatched = verification_result_factory([], as_of=different_as_of)
    result = FinalResultAssemblyService().assemble(candidate_result, mismatched)

    assert result.as_of == different_as_of  # ClaimVerificationResult wins, per design
    assert any("as_of mismatch" in note for note in result.notes)


def test_candidate_object_is_copied_verbatim_unchanged(
    candidate_result, verification_result_factory
):
    verification = verification_result_factory([])
    result = FinalResultAssemblyService().assemble(candidate_result, verification)

    for original, grounded in zip(candidate_result.candidates, result.candidates):
        assert grounded.candidate == original


def test_no_ground_truth_or_dataset_adapter_reference_in_final_result_module():
    """
    Structural guarantee, same pattern as test_claims_service.py: statically
    confirm no source file under app/services/final_result imports
    DatasetAdapter or anything from app.schemas.ground_truth, and never
    calls get_ground_truth().
    """
    import app.services.final_result.service as svc_mod

    forbidden_import_substrings = ("ground_truth", "DatasetAdapter")
    forbidden_call_names = {"get_ground_truth"}

    source = inspect.getsource(svc_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_text = ast.dump(node)
            for forbidden in forbidden_import_substrings:
                assert forbidden not in import_text, (
                    f"{svc_mod.__name__} imports something matching '{forbidden}'"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_call_names, (
                f"{svc_mod.__name__} calls forbidden function '{node.func.id}'"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_call_names, (
                f"{svc_mod.__name__} calls forbidden method '{node.func.attr}'"
            )
