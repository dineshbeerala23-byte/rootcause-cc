import pytest

from app.schemas.evidence import EvidenceBundle, EvidenceItem
from app.services.reasoning.deterministic import DeterministicCandidateGenerator
from .conftest import AS_OF, make_evidence, make_graph_with_edges, make_graph_without_edges, make_incident


def test_ranking_orders_by_aggregated_score(incident, evidence, graph_with_edges):
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph_with_edges)

    assert result.method == "deterministic"
    assert [c.service for c in result.candidates] == ["cartservice", "redis"]


def test_scores_are_summed_across_metrics_for_same_service(incident, evidence, graph_with_edges):
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph_with_edges)

    cartservice = next(c for c in result.candidates if c.service == "cartservice")
    assert cartservice.score == 9.5 + 1.0


def test_unattributed_evidence_items_are_ignored_for_ranking(incident, evidence, graph_with_edges):
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph_with_edges)

    all_supporting = {e for c in result.candidates for e in c.supporting_evidence}
    assert "unattributed_cpu" not in all_supporting


def test_supporting_evidence_lists_that_services_columns_only(incident, evidence, graph_with_edges):
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph_with_edges)

    cartservice = next(c for c in result.candidates if c.service == "cartservice")
    assert set(cartservice.supporting_evidence) == {"cartservice_cpu", "cartservice_mem"}


def test_graph_support_uses_only_real_edges(incident, evidence, graph_with_edges):
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph_with_edges)

    cartservice = next(c for c in result.candidates if c.service == "cartservice")
    # cartservice is connected to redis and checkoutservice in the fixture graph
    assert cartservice.graph_support == ["checkoutservice", "redis"]


def test_no_edges_means_graph_support_is_none_and_note_added(incident, evidence):
    graph = make_graph_without_edges(system="train-ticket")
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph)

    assert all(c.graph_support is None for c in result.candidates)
    assert any("has no edges" in note for note in result.notes)


def test_as_of_is_copied_verbatim_from_evidence_bundle(incident, evidence, graph_with_edges):
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph_with_edges)

    assert result.as_of == AS_OF == evidence.as_of


def test_no_service_attributed_evidence_yields_no_candidates(incident, graph_with_edges):
    empty_evidence = EvidenceBundle(
        case_id=incident.case_id, as_of=AS_OF, modalities_used=[], items=[]
    )
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, empty_evidence, graph_with_edges)

    assert result.candidates == []
    assert any("No service-attributed evidence" in note for note in result.notes)


def test_max_candidates_is_respected():
    incident = make_incident()
    evidence = make_evidence()
    graph = make_graph_with_edges()
    generator = DeterministicCandidateGenerator(max_candidates=1)
    result = generator.generate(incident, evidence, graph)
    assert len(result.candidates) == 1
    assert result.candidates[0].service == "cartservice"


def test_aggregation_sums_correctly_and_no_item_dominates_beyond_its_value(
    incident, graph_with_edges
):
    """
    Aggregation regression/property test, written against the current
    (unchanged) plain-sum aggregation logic in DeterministicCandidateGenerator.

    This does not test app/services/evidence/service.py's scoring formula
    directly (that has its own tests in tests/evidence/test_deviation_score.py)
    -- it tests the INTERACTION: given realistically-bounded EvidenceItem
    scores (the kind now produced after the pooled-variance correction,
    rather than the pre-fix orders-of-magnitude outliers), aggregation
    still does exactly what it claims -- sums each service's items -- and
    a service with several independently-corroborating items can correctly
    out-rank a service with one larger item, purely because the honest sum
    supports it. No single item's magnitude determines dominance beyond
    what it actually contributes to that sum.
    """
    evidence = EvidenceBundle(
        case_id=incident.case_id,
        as_of=AS_OF,
        modalities_used=["metrics"],
        items=[
            EvidenceItem(
                service="serviceA",
                metric_type="mem",
                column_name="serviceA_mem",
                summary={},
                score=120.0,
            ),
            EvidenceItem(
                service="serviceB",
                metric_type="cpu",
                column_name="serviceB_cpu",
                summary={},
                score=45.0,
            ),
            EvidenceItem(
                service="serviceB",
                metric_type="mem",
                column_name="serviceB_mem",
                summary={},
                score=40.0,
            ),
            EvidenceItem(
                service="serviceB",
                metric_type="latency-50",
                column_name="serviceB_latency-50",
                summary={},
                score=38.0,
            ),
        ],
    )
    generator = DeterministicCandidateGenerator()
    result = generator.generate(incident, evidence, graph_with_edges)

    service_a = next(c for c in result.candidates if c.service == "serviceA")
    service_b = next(c for c in result.candidates if c.service == "serviceB")

    assert service_a.score == pytest.approx(120.0)
    assert service_b.score == pytest.approx(45.0 + 40.0 + 38.0)
    # serviceB's three modest, independently-corroborated items (123.0
    # total) correctly outrank serviceA's single larger item (120.0) --
    # aggregation remains a faithful, unmodified sum; this is a property
    # check, not a magic-number tune.
    assert [c.service for c in result.candidates[:2]] == ["serviceB", "serviceA"]