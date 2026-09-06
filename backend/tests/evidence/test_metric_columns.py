import pytest

from app.services.evidence.metric_columns import parse_metric_column


@pytest.mark.parametrize(
    "column_name,expected",
    [
        ("adservice_cpu", ("adservice", "cpu")),
        ("adservice_mem", ("adservice", "mem")),
        ("adservice_load", ("adservice", "load")),
        ("adservice_latency", ("adservice", "latency")),
        ("frontend_error", ("frontend", "error")),
        ("checkoutservice_latency", ("checkoutservice", "latency")),
        ("ts-auth-service_cpu", ("ts-auth-service", "cpu")),
        ("ts-auth-service_mem", ("ts-auth-service", "mem")),
        ("ts-auth-service_latency-50", ("ts-auth-service", "latency-50")),
        ("ts-auth-service_latency-90", ("ts-auth-service", "latency-90")),
        ("ts-auth-service_workload", ("ts-auth-service", "workload")),
        ("ts-auth-service_error", ("ts-auth-service", "error")),
        # Bare "latency" suffix must still resolve on its own, not be
        # shadowed by the new latency-50/latency-90 entries.
        ("ts-auth-service_latency", ("ts-auth-service", "latency")),
        # "workload" must not be mistakenly matched by the shorter "load"
        # suffix, and vice versa "load" must still work standalone.
        ("some_service_load", ("some_service", "load")),
        ("time", (None, None)),
        ("weird_unmapped_col", (None, None)),
        ("_cpu", (None, None)),
    ],
)
def test_parse_metric_column(column_name, expected):
    assert parse_metric_column(column_name) == expected