from __future__ import annotations

from typing import Optional, Tuple

_KNOWN_METRIC_SUFFIXES = sorted(
    ["cpu", "mem", "load", "latency", "error"],
    key=len,
    reverse=True,
)


def parse_metric_column(column_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a real RCAEval wide-metrics column name of the form
    "{service}_{metric_type}" into (service, metric_type).

    Matches by known suffix, not by fixed token position, since service
    names themselves may contain underscores or hyphens (e.g.
    "ts-auth-service_cpu" -> ("ts-auth-service", "cpu")).

    Returns (None, None) if no known suffix matches -- this is a graceful
    "unparsed" result, never an exception, since the full real column
    vocabulary has not been exhaustively confirmed.
    """
    for suffix in _KNOWN_METRIC_SUFFIXES:
        marker = f"_{suffix}"
        if column_name.endswith(marker):
            service = column_name[: -len(marker)]
            if service:
                return service, suffix
    return None, None