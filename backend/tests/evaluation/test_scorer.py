from __future__ import annotations

import ast
import inspect

import pytest

from app.services.evaluation.scorer import EvaluationScorer


@pytest.fixture
def scorer() -> EvaluationScorer:
    return EvaluationScorer()


def test_top1_case_scores_correctly(scorer, final_result_top1_supported, ground_truth):
    result = scorer.score_case("case-0001", "online-boutique", final_result_top1_supported, ground_truth)

    assert result.predicted_rank == 1
    assert result.top_1_accuracy is True
    assert result.top_3_accuracy is True
    assert result.top_5_accuracy is True
    # 2 supported, 1 contradicting, 0 uncertain -> precision = 2/3, recall = 2/3
    assert result.grounding_precision == pytest.approx(2 / 3)
    assert result.grounding_recall == pytest.approx(2 / 3)
    assert result.fault_type == "cpu"
    assert result.root_cause_service == "cartservice"


def test_rank3_case_scores_correctly(scorer, final_result_rank3, ground_truth):
    result = scorer.score_case("case-0001", "online-boutique", final_result_rank3, ground_truth)

    assert result.predicted_rank == 3
    assert result.top_1_accuracy is False
    assert result.top_3_accuracy is True
    assert result.top_5_accuracy is True
    assert result.grounding_precision == pytest.approx(1.0)
    assert result.grounding_recall == pytest.approx(1.0)


def test_service_not_in_candidates_yields_none_rank_and_metrics(
    scorer, final_result_not_in_candidates, ground_truth
):
    result = scorer.score_case("case-0001", "online-boutique", final_result_not_in_candidates, ground_truth)

    assert result.predicted_rank is None
    assert result.top_1_accuracy is False
    assert result.top_3_accuracy is False
    assert result.top_5_accuracy is False
    assert result.grounding_precision is None
    assert result.grounding_recall is None
    assert any("not present among" in n for n in result.notes)


def test_case_id_mismatch_raises(scorer, final_result_top1_supported, ground_truth):
    with pytest.raises(ValueError):
        scorer.score_case("wrong-id", "online-boutique", final_result_top1_supported, ground_truth)


def test_summarize_computes_overall_and_by_system(
    scorer, final_result_top1_supported, final_result_rank3, ground_truth
):
    case_a = scorer.score_case("case-0001", "online-boutique", final_result_top1_supported, ground_truth)
    case_b = scorer.score_case("case-0001", "sock-shop", final_result_rank3, ground_truth)

    overall, by_system = scorer.summarize([case_a, case_b])

    assert overall.case_count == 2
    assert overall.top_1_accuracy == pytest.approx(0.5)
    assert overall.top_3_accuracy == pytest.approx(1.0)

    assert by_system["online-boutique"].case_count == 1
    assert by_system["online-boutique"].top_1_accuracy == pytest.approx(1.0)
    assert by_system["sock-shop"].top_1_accuracy == pytest.approx(0.0)


def test_summarize_empty_list_is_safe(scorer):
    overall, by_system = scorer.summarize([])

    assert overall.case_count == 0
    assert overall.top_1_accuracy == 0.0
    assert overall.mean_grounding_precision is None
    assert by_system == {}


def test_scorer_never_calls_get_ground_truth_or_imports_dataset_adapter():
    """
    Structural guarantee, same AST-based pattern as
    tests/claims/test_claims_service.py: EvaluationScorer legitimately
    imports the GroundTruth *type* (it reads already-fetched fields off
    it), but must never import DatasetAdapter or call get_ground_truth()
    itself -- that stays confined to EvaluationService.
    """
    import app.services.evaluation.scorer as scorer_mod

    source = inspect.getsource(scorer_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "DatasetAdapter" not in ast.dump(node), "scorer.py must not import DatasetAdapter"
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name != "get_ground_truth", "EvaluationScorer must never call get_ground_truth()"