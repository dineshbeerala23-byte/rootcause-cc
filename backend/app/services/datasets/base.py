from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.schemas.ground_truth import GroundTruth
from app.schemas.incident import CaseMetadata, DiagnosticWindow, NormalizedIncident
from app.schemas.telemetry import TelemetryQuery, TelemetrySummary


class DatasetAdapter(ABC):
    """
    Dataset-agnostic interface that every concrete dataset integration
    must implement (RCAEval first, via `RCAEvalAdapter` in Part 5;
    any future dataset later).

    Consumers above this layer (API endpoints, CLI, later phases'
    graph/verification code) depend only on this interface and on the
    Part 1 schemas it returns — never on a concrete dataset's raw file
    format or directory-naming conventions.

    Ground-truth isolation: `get_ground_truth` exists on this protocol
    because every dataset adapter needs *a* way to expose it for
    evaluation purposes, but per the Phase 3 design it must only ever
    be called from evaluation-only code paths, never from inference-
    facing code. This class does not enforce that by itself — the
    enforcement is structural (separate schema file, separate import
    path) and by caller discipline in later phases.

    Telemetry access: `get_telemetry` is intentionally left with a
    return type of the query-shape only on the input side. The lazy,
    time-cutoff-enforcing accessor object it returns (`TelemetryHandle`)
    is NOT defined yet — per the Part 1 handoff, it belongs in the
    adapter layer (Part 5), not in the pure schema layer. Concrete
    implementations will return that object; this base class does not
    reference it, so Part 2 has no dependency on code that doesn't
    exist yet.
    """

    @abstractmethod
    def list_cases(self) -> List[CaseMetadata]:
        """
        Return metadata for all cases currently available under this
        adapter's configured data source, with any dev-mode case-count
        limit already applied. Must never include ground-truth fields.
        """
        raise NotImplementedError

    @abstractmethod
    def get_incident(self, case_id: str) -> NormalizedIncident:
        """
        Return the normalized, ground-truth-free incident for the given
        opaque case_id. Must raise a dataset-layer exception (not a bare
        KeyError/FileNotFoundError) if case_id is unknown or the
        underlying case data is missing/corrupt.
        """
        raise NotImplementedError

    @abstractmethod
    def get_telemetry_summary(
        self, case_id: str, window: DiagnosticWindow
    ) -> TelemetrySummary:
        """
        Return a summary of what telemetry is available for this case
        within the given diagnostic window (modalities present, row
        counts, etc.), without returning the raw rows themselves.
        """
        raise NotImplementedError

    @abstractmethod
    def get_telemetry(self, case_id: str, query: TelemetryQuery):
        """
        Return a lazy telemetry accessor for this case, scoped to
        `query` (modality/service filters, time window). Implementations
        must enforce the `as_of` cutoff internally — callers should
        never receive rows beyond `as_of`, and `as_of=None` must resolve
        to `inject_time + small fixed buffer`, never to "everything in
        the file." Concrete return type is defined in Part 5.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self, case_id: str) -> GroundTruth:
        """
        Evaluation-only accessor. Returns the real fault type / faulted
        service for this case. Must only be called from evaluation-layer
        code — never from inference-facing paths, prompts, or default
        API response bodies.
        """
        raise NotImplementedError