"""Future-data cutoff enforcement tests."""

from app.services.datasets.rcaeval.adapter import RCAEvalAdapter
from app.services.datasets.rcaeval.discovery import RCAEvalDiscovery
from app.schemas.telemetry import TelemetryQuery


def test_rows_after_as_of_are_never_returned(rcaeval_settings):
    adapter = RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))
    cases = {c.system: c.case_id for c in adapter.list_cases()}
    case_id = cases["online-boutique"]

    handle = adapter.get_telemetry(case_id, TelemetryQuery())
    metrics_df = handle.metrics()

    # inject_time = 1700000000, default buffer = 5 min -> cutoff = 1700000300
    assert (metrics_df["time"] <= 1700000300).all()
    assert 1700100000 not in metrics_df["time"].values


def test_telemetry_summary_time_range_respects_cutoff(rcaeval_settings):
    adapter = RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))
    cases = {c.system: c.case_id for c in adapter.list_cases()}
    case_id = cases["sock-shop"]

    incident = adapter.get_incident(case_id)
    summary = adapter.get_telemetry_summary(case_id, incident.diagnostic_window)

    assert summary.time_range.end is not None
    assert summary.time_range.end.timestamp() <= 1700000300


def test_logs_and_traces_after_cutoff_excluded(rcaeval_settings):
    adapter = RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))
    cases = {c.system: c.case_id for c in adapter.list_cases()}
    case_id = cases["sock-shop"]

    handle = adapter.get_telemetry(case_id, TelemetryQuery())
    logs_df = handle.logs()

    assert "elevated response time observed - excluded from any diagnostic window" not in (
        logs_df["message"].tolist() if "message" in logs_df.columns else []
    )