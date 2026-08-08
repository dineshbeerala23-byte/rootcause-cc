"""Development case-count limit tests."""

from app.core.config import Settings
from app.services.datasets.rcaeval.discovery import RCAEvalDiscovery


def test_no_limit_returns_all_cases(rcaeval_settings):
    discovery = RCAEvalDiscovery(settings=rcaeval_settings)
    assert len(discovery.list_cases()) == 2


def test_dev_mode_setting_caps_results(rcaeval_fixture_dir):
    settings = Settings(
        RCAEVAL_DATA_DIR=str(rcaeval_fixture_dir),
        RCAEVAL_DEV_MODE_MAX_CASES=1,
    )
    discovery = RCAEvalDiscovery(settings=settings)
    cases = discovery.list_cases()

    assert len(cases) == 1
    assert cases[0].case_id == "case-0001"  # deterministic: sorted first


def test_explicit_max_cases_overrides_setting(rcaeval_fixture_dir):
    settings = Settings(
        RCAEVAL_DATA_DIR=str(rcaeval_fixture_dir),
        RCAEVAL_DEV_MODE_MAX_CASES=None,
    )
    discovery = RCAEvalDiscovery(settings=settings)

    assert len(discovery.list_cases(max_cases=1)) == 1
    assert len(discovery.list_cases()) == 2