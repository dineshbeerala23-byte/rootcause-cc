from __future__ import annotations

import ast
import inspect

import pytest

from app.schemas.claims import Claim, ClaimGenerationResult
from app.services.claims.base import ClaimGenerator, ClaimVerifier
from app.services.claims.service import ClaimVerificationService
from app.services.claims.verifier import DeterministicClaimVerifier


class _StubGenerator(ClaimGenerator):
    """Returns a fixed ClaimGenerationResult, bypassing the LLM entirely."""

    def __init__(self, result: ClaimGenerationResult):
        self._result = result

    async def generate(self, candidate_result, evidence, graph):
        return self._result


@pytest.mark.asyncio
async def test_generator_output_flows_into_verifier(candidate_result, evidence_bundle, service_graph):
    generation_result = ClaimGenerationResult(
        case_id=candidate_result.case_id,
        as_of=evidence_bundle.as_of,
        method="llm-generated",
        claims=[
            Claim(
                claim_id="c1",
                candidate_service="cartservice",
                statement="cartservice shows elevated CPU and latency.",
                cited_evidence=["cartservice_cpu", "cartservice_latency"],
            )
        ],
        notes=[],
    )
    service = ClaimVerificationService(claim_generator=_StubGenerator(generation_result))

    result = await service.analyze(candidate_result, evidence_bundle, service_graph)

    assert len(result.verdicts) == 1
    assert result.verdicts[0].status == "supported"
    assert result.as_of == evidence_bundle.as_of


@pytest.mark.asyncio
async def test_empty_generation_result_still_produces_valid_empty_verification(
    candidate_result, evidence_bundle, service_graph
):
    empty_generation_result = ClaimGenerationResult(
        case_id=candidate_result.case_id,
        as_of=evidence_bundle.as_of,
        method="llm-generated",
        claims=[],
        notes=["LLM claim generation failed: provider unavailable"],
    )
    service = ClaimVerificationService(claim_generator=_StubGenerator(empty_generation_result))

    result = await service.analyze(candidate_result, evidence_bundle, service_graph)

    assert result.verdicts == []
    assert any("provider unavailable" in note for note in result.notes)


@pytest.mark.asyncio
async def test_default_verifier_is_deterministic_claim_verifier(candidate_result, evidence_bundle, service_graph):
    generation_result = ClaimGenerationResult(
        case_id=candidate_result.case_id,
        as_of=evidence_bundle.as_of,
        method="llm-generated",
        claims=[],
        notes=[],
    )
    service = ClaimVerificationService(claim_generator=_StubGenerator(generation_result))

    assert isinstance(service._claim_verifier, DeterministicClaimVerifier)


def test_no_dataset_adapter_or_ground_truth_reference_anywhere_in_claims_module():
    """
    Structural guarantee, mirroring RCAReasoningService: statically confirm
    that no source file under app/services/claims imports DatasetAdapter or
    anything from app.schemas.ground_truth, and never calls
    get_ground_truth(). This is checked via AST inspection of the actual
    module source, not by trusting docstrings.
    """
    import app.services.claims.base as base_mod
    import app.services.claims.llm_generator as gen_mod
    import app.services.claims.service as svc_mod
    import app.services.claims.verifier as ver_mod

    forbidden_import_substrings = ("ground_truth", "DatasetAdapter")
    forbidden_call_names = {"get_ground_truth"}

    for module in (base_mod, gen_mod, svc_mod, ver_mod):
        source = inspect.getsource(module)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_text = ast.dump(node)
                for forbidden in forbidden_import_substrings:
                    assert forbidden not in import_text, (
                        f"{module.__name__} imports something matching '{forbidden}'"
                    )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_call_names, (
                    f"{module.__name__} calls forbidden function '{node.func.id}'"
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_call_names, (
                    f"{module.__name__} calls forbidden method '{node.func.attr}'"
                )
