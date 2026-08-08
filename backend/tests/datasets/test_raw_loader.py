"""Tests for app.services.datasets.rcaeval.raw_loader.RCAEvalRawCase."""

import pytest

from app.core.exceptions import CorruptDatasetFileError, DatasetCaseNotFoundError
from app.services.datasets.rcaeval.raw_loader import RCAEvalRawCase


def test_missing_case_directory_raises_not_found(tmp_path):
    with pytest.raises(DatasetCaseNotFoundError):
        RCAEvalRawCase(tmp_path / "does_not_exist")


def test_load_metrics_from_real_fixture(rcaeval_fixture_dir):
    case_dir = rcaeval_fixture_dir / "RE1" / "online-boutique_cartservice_cpu_1"
    case = RCAEvalRawCase(case_dir)

    assert case.has_metrics()
    assert not case.has_logs()
    assert not case.has_traces()

    df = case.load_metrics()
    assert "time" in df.columns
    assert len(df) == 3

    # cached: second call returns the same object, not re-read from disk
    assert case.load_metrics() is df


def test_load_inject_time(rcaeval_fixture_dir):
    case_dir = rcaeval_fixture_dir / "RE1" / "online-boutique_cartservice_cpu_1"
    case = RCAEvalRawCase(case_dir)
    assert case.load_inject_time() == 1700000000.0


def test_multi_modality_case_loads_logs_and_traces(rcaeval_fixture_dir):
    case_dir = rcaeval_fixture_dir / "RE2" / "sock-shop_orders_delay_2"
    case = RCAEvalRawCase(case_dir)

    assert case.has_metrics() and case.has_logs() and case.has_traces()
    assert len(case.load_logs()) == 3
    assert len(case.load_traces()) == 3


def test_missing_metrics_file_raises_corrupt_error(tmp_path):
    case_dir = tmp_path / "empty_case"
    case_dir.mkdir()
    case = RCAEvalRawCase(case_dir)

    with pytest.raises(CorruptDatasetFileError):
        case.load_metrics()


def test_malformed_metrics_json_raises_corrupt_error(tmp_path):
    case_dir = tmp_path / "bad_case"
    case_dir.mkdir()
    (case_dir / "metrics.json").write_text("not valid json {{{")

    case = RCAEvalRawCase(case_dir)
    with pytest.raises(CorruptDatasetFileError):
        case.load_metrics()


def test_metrics_json_missing_time_column_raises_corrupt_error(tmp_path):
    case_dir = tmp_path / "no_time_case"
    case_dir.mkdir()
    (case_dir / "metrics.json").write_text('[{"cpu_usage_pct": 1.0}]')

    case = RCAEvalRawCase(case_dir)
    with pytest.raises(CorruptDatasetFileError):
        case.load_metrics()


def test_missing_inject_time_raises_corrupt_error(tmp_path):
    case_dir = tmp_path / "no_inject_time_case"
    case_dir.mkdir()
    (case_dir / "metrics.json").write_text('[{"time": 1, "cpu_usage_pct": 1.0}]')

    case = RCAEvalRawCase(case_dir)
    with pytest.raises(CorruptDatasetFileError):
        case.load_inject_time()