"""Tests for app.services.datasets.rcaeval.adapter.RCAEvalAdapter."""

from app.services.datasets.rcaeval.adapter import RCAEvalAdapter
from app.services.datasets.rcaeval.discovery import RCAEvalDiscovery
from app.schemas.telemetry import TelemetryQuery


def _adapter(rcaeval_settings) -> RCAEvalAdapter:
    return RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))


def test_get_incident_returns_normalized_incident(rcaeval_settings):
    adapter = _adapter(rcaeval_settings)
    case_id = adapter.list_cases()[0].case_id

    incident = adapter.get_incident(case_id)
    assert incident.case_id == case_id
    assert incident.diagnostic_window.inject_time is not None


def test_get_telemetry_summary_is_bounded_and_filtered(rcaeval_settings):
    adapter = _adapter(rcaeval_settings)
    cases = {c.system: c.case_id for c in adapter.list_cases()}
    case_id = cases["online-boutique"]

    incident = adapter.get_incident(case_id)
    summary = adapter.get_telemetry_summary(case_id, incident.diagnostic_window)

    assert summary.case_id == case_id
    assert summary.modalities.metrics is True
    # 3 rows exist; only 2 are within inject_time + 5min buffer
    assert summary.row_counts["metrics"] == 2


def test_get_telemetry_filters_by_service_and_max_rows(rcaeval_settings):
    adapter = _adapter(rcaeval_settings)
    cases = {c.system: c.case_id for c in adapter.list_cases()}
    case_id = cases["sock-shop"]

    handle = adapter.get_telemetry(case_id, TelemetryQuery(service="orders", max_rows=1))
    metrics_df = handle.metrics()

    assert len(metrics_df) == 1
    assert (metrics_df["service"] == "orders").all()


def test_get_telemetry_multi_modality(rcaeval_settings):
    adapter = _adapter(rcaeval_settings)
    cases = {c.system: c.case_id for c in adapter.list_cases()}
    case_id = cases["sock-shop"]

    handle = adapter.get_telemetry(case_id, TelemetryQuery())
    assert len(handle.metrics()) == 2  # 2 of 3 rows within cutoff
    assert len(handle.logs()) == 2
    assert len(handle.traces()) == 2