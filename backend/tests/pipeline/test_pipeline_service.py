from __future__ import annotations

import ast
import inspect

import pytest


@pytest.mark.asyncio
async def test_analyze_calls_full_sequence_and_returns_expected_shape(pipeline_service_factory):
    service, fakes = pipeline_service_factory()

    response = await service.analyze("case-0001", mode="deterministic")

    assert response.case_id == "case-0001"
    assert response.mode == "deterministic"
    assert response.provider == "fake-provider"
    assert response.model == "fake-model"
    assert response.result.case_id == "case-0001"
    assert len(response.result.candidates) == 1
    assert response.result.candidates[0].grounding_status == "supported"

    assert fakes["adapter"].last_case_id == "case-0001"
    assert fakes["evidence_service"].last_case_id == "case-0001"
    assert fakes["topology_provider"].last_system == "online-boutique"
    assert fakes["reasoning_service"].last_mode == "deterministic"


@pytest.mark.asyncio
async def test_mode_is_passed_through_to_reasoning_service(pipeline_service_factory):
    service, fakes = pipeline_service_factory()

    await service.analyze("case-0001", mode="llm-assisted")

    assert fakes["reasoning_service"].last_mode == "llm-assisted"


@pytest.mark.asyncio
async def test_case_id_is_taken_from_resolved_incident(pipeline_service_factory):
    service, fakes = pipeline_service_factory()

    response = await service.analyze("case-0001")

    assert fakes["adapter"].last_case_id == "case-0001"
    assert response.case_id == "case-0001"


def test_no_ground_truth_or_get_ground_truth_reference_in_pipeline_service_module():
    """
    Structural guarantee, same AST-based pattern as
    tests/claims/test_claims_service.py and
    tests/final_result/test_final_result_service.py: confirm the pipeline
    service module never imports app.schemas.ground_truth and never calls
    get_ground_truth().
    """
    import app.services.pipeline.service as svc_mod

    forbidden_import_substrings = ("ground_truth",)
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
