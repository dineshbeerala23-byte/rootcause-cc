from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from app.schemas.evidence import EvidenceBundle, EvidenceItem, EvidenceQuery
from app.schemas.telemetry import TelemetryQuery
from app.services.datasets.base import DatasetAdapter
from app.services.evidence.metric_columns import parse_metric_column

_EPSILON = 1e-9


def _deviation_score(series: pd.Series) -> float:
    """
    "How much did this signal move" score -- a signal-movement/change
    ranking heuristic ONLY, not an RCA or root-cause detector. Compares
    the mean of the last quarter of the given series against the mean
    of the first quarter, normalized by a POOLED standard deviation
    across both the early and late windows. Never looks outside the
    series it's handed.

    Pooled (not early-window-only) variance is used deliberately: an
    early-window-only denominator collapses toward the epsilon floor
    whenever the early quarter happens to be flat/near-constant (e.g.
    quantized baseline memory readings), producing scores many orders
    of magnitude larger than any other metric purely as an artifact of
    which quarter was used as "baseline" -- not because of a stronger
    underlying signal. Pooling both windows only produces a near-zero
    denominator when the ENTIRE compared span is flat, which is a much
    rarer and more legitimately near-zero-signal condition.
    """
    n = len(series)
    if n == 0:
        return 0.0
    if n < 4:
        return float(abs(series.iloc[-1] - series.iloc[0]))

    quarter = max(1, n // 4)
    early = series.iloc[:quarter]
    late = series.iloc[-quarter:]

    early_mean = early.mean()
    late_mean = late.mean()
    early_std = early.std(ddof=0)
    late_std = late.std(ddof=0)
    if pd.isna(early_std):
        early_std = 0.0
    if pd.isna(late_std):
        late_std = 0.0

    mean_shift = abs(late_mean - early_mean)

    if early_std == 0.0 and late_std == 0.0:
        # No variability in EITHER window to normalize against at all --
        # this is the flat-to-flat regime (e.g. a quantized metric moving
        # from one exact plateau to another with zero jitter in both
        # windows). Dividing by an arbitrary epsilon here has no
        # principled scale and was the source of the observed
        # trillions-scale scores. Fall back to a relative (magnitude-
        # normalized) change measure instead: mean shift as a fraction
        # of the larger window's own level. This is bounded, dimensionless,
        # and comparable across metric types, the same role the pooled-std
        # ratio plays when real variance exists -- it's just normalized by
        # signal level instead of signal noise, since no noise floor
        # exists in this branch. When exactly one window has zero
        # variance (not both), no special case is needed: the pooled
        # formula below already stays well-conditioned in that case,
        # since pooled_std reduces to nonzero_std/sqrt(2).
        scale = max(abs(early_mean), abs(late_mean))
        return float(mean_shift / (scale + _EPSILON))

    pooled_std = ((early_std**2 + late_std**2) / 2) ** 0.5

    return float(mean_shift / (pooled_std + _EPSILON))

class EvidenceRetrievalService:
    """
    Turns bounded, cutoff-safe telemetry (via DatasetAdapter) into a
    small, ranked EvidenceBundle. Depends only on DatasetAdapter's
    public interface -- never on RCAEvalAdapter, RCAEvalDiscovery, or
    any raw file path.

    v1 scope: metrics modality only (wide-format columns), simple
    descriptive-stat ranking. Explicitly NOT an RCA/root-cause detector
    -- see _deviation_score's docstring.
    """

    def __init__(self, adapter: DatasetAdapter):
        self._adapter = adapter

    def get_evidence(self, case_id: str, query: EvidenceQuery) -> EvidenceBundle:
        incident = self._adapter.get_incident(case_id)

        telemetry_query = TelemetryQuery(service=None, window=query.window, max_rows=None)
        handle = self._adapter.get_telemetry(case_id, telemetry_query)

        modalities_used: List[str] = []
        items: List[EvidenceItem] = []
        observed_frames: List[pd.DataFrame] = []

        if incident.metadata.available_modalities.metrics:
            metrics_df = handle.metrics()
            if not metrics_df.empty:
                modalities_used.append("metrics")
                observed_frames.append(metrics_df)
                items.extend(self._score_metrics(metrics_df, query))

        # Logs/traces intentionally not handled -- no real RE2/RE3 data
        # exists yet to build/validate a strategy against.

        items.sort(key=lambda item: (-item.score, item.column_name))
        bounded_items = items[: query.max_items]

        as_of = self._resolve_as_of(observed_frames, incident.diagnostic_window.inject_time)

        return EvidenceBundle(
            case_id=case_id,
            as_of=as_of,
            modalities_used=modalities_used,
            items=bounded_items,
        )

    def _score_metrics(self, df: pd.DataFrame, query: EvidenceQuery) -> List[EvidenceItem]:
        results: List[EvidenceItem] = []
        for column in df.columns:
            if column == "time":
                continue

            service, metric_type = parse_metric_column(column)

            if query.services is not None and service not in query.services:
                continue
            if query.metric_types is not None and metric_type not in query.metric_types:
                continue

            series = df[column].dropna()
            if series.empty:
                continue

            summary = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0) or 0.0),
                "min": float(series.min()),
                "max": float(series.max()),
                "last": float(series.iloc[-1]),
            }
            score = _deviation_score(series)

            results.append(
                EvidenceItem(
                    service=service,
                    metric_type=metric_type,
                    column_name=column,
                    summary=summary,
                    score=score,
                )
            )
        return results

    @staticmethod
    def _resolve_as_of(
        observed_frames: List[pd.DataFrame], inject_time: datetime
    ) -> datetime:
        max_time: Optional[float] = None
        for frame in observed_frames:
            if frame.empty:
                continue
            local_max = frame["time"].max()
            max_time = local_max if max_time is None else max(max_time, local_max)

        if max_time is None:
            return inject_time
        return datetime.fromtimestamp(max_time, tz=timezone.utc)