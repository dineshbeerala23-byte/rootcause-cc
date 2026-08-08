from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, get_args

import pandas as pd

from app.core.config import Settings
from app.core.exceptions import DatasetCaseNotFoundError, DatasetError
from app.schemas.ground_truth import GroundTruth
from app.schemas.incident import CaseMetadata, DatasetSuite, DiagnosticWindow, NormalizedIncident
from app.schemas.telemetry import (
    ModalityAvailability,
    TelemetryQuery,
    TelemetrySummary,
    TimeWindow,
)
from app.services.datasets.base import DatasetAdapter
from app.services.datasets.rcaeval.discovery import RCAEvalDiscovery, _parse_system_from_dir_name
from app.services.datasets.rcaeval.raw_loader import RCAEvalRawCase

_KNOWN_SUITES = get_args(DatasetSuite)


class TelemetryHandle:
    """
    Lazy, cutoff-enforcing telemetry accessor returned by
    RCAEvalAdapter.get_telemetry(). No accessor method here can ever
    return a row beyond the resolved as_of cutoff — filtering is applied
    on every call, not cached from construction time, so there is no
    code path that returns unfiltered data.

    'time' columns in RCAEval's raw files are assumed to be numeric Unix
    timestamps (matching inject_time.txt's format) — the same assumption
    the raw loader (Part 3) and discovery (Part 4) already rely on
    implicitly by treating 'time' as a plain comparable column.
    """

    def __init__(
        self,
        raw_case: RCAEvalRawCase,
        as_of: datetime,
        service: Optional[str],
        max_rows: Optional[int],
        window: Optional[TimeWindow],
    ):
        self._raw_case = raw_case
        self._as_of = as_of
        self._service = service
        self._max_rows = max_rows
        self._window = window

    def _effective_end(self) -> datetime:
        # The query's optional window can only narrow inward; it can
        # never push the cutoff later than the resolved as_of.
        if self._window is not None and self._window.end is not None:
            return min(self._window.end, self._as_of)
        return self._as_of

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        end = self._effective_end()
        mask = df["time"] <= end.timestamp()

        if self._window is not None and self._window.start is not None:
            mask &= df["time"] >= self._window.start.timestamp()

        result = df[mask]

        if self._service is not None and "service" in result.columns:
            result = result[result["service"] == self._service]

        if self._max_rows is not None:
            result = result.head(self._max_rows)

        return result

    def metrics(self) -> pd.DataFrame:
        if not self._raw_case.has_metrics():
            return pd.DataFrame(columns=["time"])
        return self._apply(self._raw_case.load_metrics())

    def logs(self) -> pd.DataFrame:
        if not self._raw_case.has_logs():
            return pd.DataFrame(columns=["time"])
        return self._apply(self._raw_case.load_logs())

    def traces(self) -> pd.DataFrame:
        if not self._raw_case.has_traces():
            return pd.DataFrame(columns=["time"])
        return self._apply(self._raw_case.load_traces())


class RCAEvalAdapter(DatasetAdapter):
    """
    Wires discovery (Part 4), the raw loader (Part 3), and the Part 1
    schemas together into the DatasetAdapter interface (Part 2).

    Every method here is the boundary that must never leak a raw
    RCAEval directory name or ground-truth information into an
    inference-facing return value. Only get_ground_truth is allowed to
    surface the answer, and only because that is its sole documented
    purpose — callers are responsible for only invoking it from
    evaluation-only code, per the Part 2/base.py contract.
    """

    _AS_OF_DEFAULT_BUFFER = timedelta(minutes=5)
    _DEFAULT_SAMPLE_ROWS = 20

    def __init__(
        self,
        discovery: Optional[RCAEvalDiscovery] = None,
        settings: Optional[Settings] = None,
    ):
        self._discovery = discovery or RCAEvalDiscovery(settings=settings)

    # --- internal helpers -------------------------------------------------

    def _raw_case_for(self, case_id: str) -> RCAEvalRawCase:
        case_dir = self._discovery.get_case_dir(case_id)
        return RCAEvalRawCase(case_dir)

    def _resolve_as_of(self, window: DiagnosticWindow) -> datetime:
        if window.as_of is not None:
            return window.as_of
        return window.inject_time + self._AS_OF_DEFAULT_BUFFER

    @staticmethod
    def _filter_by_cutoff(df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
        return df[df["time"] <= as_of.timestamp()]

    # --- DatasetAdapter interface ------------------------------------------

    def list_cases(self) -> List[CaseMetadata]:
        return self._discovery.list_cases()

    def get_incident(self, case_id: str) -> NormalizedIncident:
        case_dir: Path = self._discovery.get_case_dir(case_id)
        raw_case = RCAEvalRawCase(case_dir)

        suite = case_dir.parent.name
        if suite not in _KNOWN_SUITES:
            raise DatasetError(
                f"Unrecognized suite folder {suite!r} for case_id={case_id!r}."
            )

        system = _parse_system_from_dir_name(case_dir.name)

        metadata = CaseMetadata(
            case_id=case_id,
            dataset_suite=suite,  # type: ignore[arg-type]
            system=system,
            available_modalities=ModalityAvailability(
                metrics=raw_case.has_metrics(),
                logs=raw_case.has_logs(),
                traces=raw_case.has_traces(),
            ),
        )

        inject_time_dt = datetime.fromtimestamp(raw_case.load_inject_time(), tz=timezone.utc)
        diagnostic_window = DiagnosticWindow(inject_time=inject_time_dt, as_of=None)

        return NormalizedIncident(metadata=metadata, diagnostic_window=diagnostic_window)

    def get_telemetry_summary(
        self, case_id: str, window: DiagnosticWindow
    ) -> TelemetrySummary:
        raw_case = self._raw_case_for(case_id)
        as_of = self._resolve_as_of(window)

        modalities = ModalityAvailability(
            metrics=raw_case.has_metrics(),
            logs=raw_case.has_logs(),
            traces=raw_case.has_traces(),
        )

        row_counts: Dict[str, int] = {}
        sample: Dict[str, list] = {}
        service_names: set = set()
        min_time: Optional[float] = None
        max_time: Optional[float] = None

        modality_loaders = (
            ("metrics", modalities.metrics, raw_case.load_metrics),
            ("logs", modalities.logs, raw_case.load_logs),
            ("traces", modalities.traces, raw_case.load_traces),
        )

        for name, available, loader in modality_loaders:
            if not available:
                continue

            filtered = self._filter_by_cutoff(loader(), as_of)
            row_counts[name] = len(filtered)

            if "service" in filtered.columns:
                service_names.update(filtered["service"].dropna().unique().tolist())

            if not filtered.empty:
                local_min = filtered["time"].min()
                local_max = filtered["time"].max()
                min_time = local_min if min_time is None else min(min_time, local_min)
                max_time = local_max if max_time is None else max(max_time, local_max)

            sample[name] = filtered.head(self._DEFAULT_SAMPLE_ROWS).to_dict(orient="records")

        time_range = TimeWindow(
            start=datetime.fromtimestamp(min_time, tz=timezone.utc) if min_time is not None else None,
            end=datetime.fromtimestamp(max_time, tz=timezone.utc) if max_time is not None else None,
        )

        return TelemetrySummary(
            case_id=case_id,
            modalities=modalities,
            service_names=sorted(service_names),
            row_counts=row_counts,
            time_range=time_range,
            sample=sample,
        )

    def get_telemetry(self, case_id: str, query: TelemetryQuery) -> TelemetryHandle:
        raw_case = self._raw_case_for(case_id)

        # get_telemetry's signature (fixed by Part 2's approved
        # DatasetAdapter interface) takes no DiagnosticWindow, so the
        # cutoff is resolved fresh here from inject_time + the default
        # buffer rather than from any previously-resolved as_of a
        # caller might already hold via get_incident/get_telemetry_summary.
        inject_time_dt = datetime.fromtimestamp(raw_case.load_inject_time(), tz=timezone.utc)
        as_of = inject_time_dt + self._AS_OF_DEFAULT_BUFFER

        return TelemetryHandle(
            raw_case=raw_case,
            as_of=as_of,
            service=query.service,
            max_rows=query.max_rows,
            window=query.window,
        )

    def get_ground_truth(self, case_id: str) -> GroundTruth:
        case_dir = self._discovery.get_case_dir(case_id)
        real_name = case_dir.name

        parts = real_name.split("_")
        if len(parts) < 4:
            raise DatasetError(
                f"Cannot parse ground truth for case_id={case_id!r}: "
                f"directory name {real_name!r} does not match "
                f"'{{system}}_{{service}}_{{fault}}_{{instance}}'."
            )

        fault_type = parts[-2]
        service = "_".join(parts[1:-2])

        if not service:
            raise DatasetError(
                f"Cannot parse root_cause_service for case_id={case_id!r}: "
                f"directory name {real_name!r} yielded an empty service token."
            )

        return GroundTruth(
            case_id=case_id,
            root_cause_service=service,
            fault_type=fault_type,
            root_cause_indicator=None,
        )