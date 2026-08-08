"""
Explicit leakage tests: ground-truth information must never appear in
inference-facing objects, only through get_ground_truth().
"""

from app.schemas.incident import CaseMetadata, NormalizedIncident
from app.services.datasets.rcaeval.adapter import RCAEvalAdapter
from app.services.datasets.rcaeval.discovery import RCAEvalDiscovery


def test_normalized_incident_and_case_metadata_have_no_ground_truth_fields():
    forbidden_fields = {"root_cause_service", "fault_type", "root_cause_indicator"}
    assert forbidden_fields.isdisjoint(NormalizedIncident.model_fields.keys())
    assert forbidden_fields.isdisjoint(CaseMetadata.model_fields.keys())


def test_get_ground_truth_returns_correct_answer(rcaeval_settings):
    adapter = RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))
    cases = {c.system: c.case_id for c in adapter.list_cases()}

    gt_cart = adapter.get_ground_truth(cases["online-boutique"])
    assert gt_cart.root_cause_service == "cartservice"
    assert gt_cart.fault_type == "cpu"

    gt_orders = adapter.get_ground_truth(cases["sock-shop"])
    assert gt_orders.root_cause_service == "orders"
    assert gt_orders.fault_type == "delay"


def test_incident_serialization_never_contains_ground_truth_tokens(rcaeval_settings):
    adapter = RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))
    cases = {c.system: c.case_id for c in adapter.list_cases()}
    case_id = cases["online-boutique"]

    incident = adapter.get_incident(case_id)
    dumped = incident.model_dump_json()

    assert "cartservice" not in dumped
    assert "cpu" not in dumped
    assert "online-boutique_cartservice_cpu_1" not in dumped


def test_case_metadata_serialization_never_contains_ground_truth_tokens(rcaeval_settings):
    adapter = RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))
    cases = adapter.list_cases()

    for case in cases:
        dumped = case.model_dump_json()
        assert "cartservice" not in dumped
        assert "orders" not in dumped
        assert "cpu" not in dumped
        assert "delay" not in dumped