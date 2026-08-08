"""Tests for app.services.datasets.rcaeval.discovery.RCAEvalDiscovery."""

import pytest

from app.core.config import Settings
from app.core.exceptions import DatasetNotConfiguredError
from app.services.datasets.rcaeval.discovery import RCAEvalDiscovery


def test_list_cases_finds_both_fixture_cases(rcaeval_settings):
    discovery = RCAEvalDiscovery(settings=rcaeval_settings)
    cases = discovery.list_cases()

    assert len(cases) == 2
    assert {c.system for c in cases} == {"online-boutique", "sock-shop"}
    assert {c.dataset_suite for c in cases} == {"RE1", "RE2"}
    # opaque ids only, sorted
    assert [c.case_id for c in cases] == ["case-0001", "case-0002"]


def test_list_cases_never_exposes_raw_directory_name(rcaeval_settings):
    discovery = RCAEvalDiscovery(settings=rcaeval_settings)
    cases = discovery.list_cases()

    for case in cases:
        dumped = case.model_dump_json()
        assert "cartservice" not in dumped
        assert "orders" not in dumped
        assert "_cpu_" not in dumped
        assert "_delay_" not in dumped


def test_filter_by_suite_and_system(rcaeval_settings):
    discovery = RCAEvalDiscovery(settings=rcaeval_settings)

    re1_cases = discovery.list_cases(suite="RE1")
    assert len(re1_cases) == 1
    assert re1_cases[0].system == "online-boutique"

    sock_shop_cases = discovery.list_cases(system="sock-shop")
    assert len(sock_shop_cases) == 1
    assert sock_shop_cases[0].dataset_suite == "RE2"


def test_missing_data_dir_raises_not_configured(tmp_path):
    settings = Settings(RCAEVAL_DATA_DIR=str(tmp_path / "does_not_exist"))
    discovery = RCAEvalDiscovery(settings=settings)

    with pytest.raises(DatasetNotConfiguredError):
        discovery.list_cases()


def test_empty_data_dir_raises_not_configured(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    settings = Settings(RCAEVAL_DATA_DIR=str(empty_dir))
    discovery = RCAEvalDiscovery(settings=settings)

    with pytest.raises(DatasetNotConfiguredError):
        discovery.list_cases()


def test_get_case_dir_resolves_to_real_directory(rcaeval_settings, rcaeval_fixture_dir):
    discovery = RCAEvalDiscovery(settings=rcaeval_settings)
    cases = discovery.list_cases(suite="RE1")
    case_dir = discovery.get_case_dir(cases[0].case_id)

    assert case_dir == rcaeval_fixture_dir / "RE1" / "online-boutique_cartservice_cpu_1"