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
        ("time", (None, None)),
        ("weird_unmapped_col", (None, None)),
        ("_cpu", (None, None)),
    ],
)
def test_parse_metric_column(column_name, expected):
    assert parse_metric_column(column_name) == expected