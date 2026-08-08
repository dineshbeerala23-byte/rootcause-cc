"""Tests for app.services.datasets.rcaeval.case_id_registry.CaseIdRegistry."""

import pytest

from app.core.exceptions import DatasetCaseNotFoundError
from app.services.datasets.rcaeval.case_id_registry import CaseIdRegistry


def test_deterministic_sorted_assignment():
    registry = CaseIdRegistry(
        ["sock-shop_orders_delay_2", "online-boutique_cartservice_cpu_1"]
    )
    assert registry.all_case_ids() == ["case-0001", "case-0002"]
    # sorted alphabetically: "online-boutique..." < "sock-shop..."
    assert registry.opaque_id_for("online-boutique_cartservice_cpu_1") == "case-0001"
    assert registry.opaque_id_for("sock-shop_orders_delay_2") == "case-0002"


def test_round_trip_lookup():
    registry = CaseIdRegistry(["online-boutique_cartservice_cpu_1"])
    opaque = registry.opaque_id_for("online-boutique_cartservice_cpu_1")
    assert registry.real_directory_name_for(opaque) == "online-boutique_cartservice_cpu_1"


def test_duplicate_real_names_deduplicated():
    registry = CaseIdRegistry(["dup_a_b_1", "dup_a_b_1", "dup_a_b_1"])
    assert len(registry.all_case_ids()) == 1


def test_unknown_real_name_raises_not_found():
    registry = CaseIdRegistry(["online-boutique_cartservice_cpu_1"])
    with pytest.raises(DatasetCaseNotFoundError):
        registry.opaque_id_for("never_seen_directory")


def test_unknown_opaque_id_raises_not_found():
    registry = CaseIdRegistry(["online-boutique_cartservice_cpu_1"])
    with pytest.raises(DatasetCaseNotFoundError):
        registry.real_directory_name_for("case-9999")