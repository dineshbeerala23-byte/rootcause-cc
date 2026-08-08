# ============================================================
# app/schemas/telemetry.py  (NEW)
# ============================================================
"""
Telemetry-related data-transfer schemas.

These are plain, serializable shapes for describing *what telemetry is
available* and *what window/filter was applied* — not the telemetry data
itself. Raw metrics/logs/traces stay as pandas DataFrames inside the
adapter layer (Part 5) and are never sent to the LLM or the API wholesale;
only bounded TelemetrySummary objects (defined here) cross that boundary.

The lazy-loading TelemetryHandle class (which actually calls into
raw_loader.py to fetch pandas data) is NOT defined here — it depends on
raw_loader.py, which doesn't exist until Part 3, and it isn't a pure data
schema. It's implemented in Part 5 (app/services/datasets/rcaeval/adapter.py).
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Modality = Literal["metrics", "logs", "traces"]


class TimeWindow(BaseModel):
    """
    An optional [start, end] bound. Both ends optional; None means
    unbounded on that side. Used both as an incoming filter (API/CLI
    callers asking for a slice) and as a descriptive field (what range a
    TelemetrySummary actually covers).
    """

    model_config = ConfigDict(extra="forbid")

    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @model_validator(mode="after")
    def _check_order(self) -> "TimeWindow":
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("TimeWindow.start must not be after TimeWindow.end")
        return self


class TelemetryQuery(BaseModel):
    """
    What a caller (API endpoint, CLI command, future RCA method) asks for
    when requesting telemetry for one case. Deliberately does NOT include
    an 'as_of' override that could bypass a case's diagnostic window — the
    diagnostic-window cutoff is enforced by DiagnosticWindow (incident.py)
    and applied inside the adapter, not negotiable per-query.
    """

    model_config = ConfigDict(extra="forbid")

    service: Optional[str] = Field(
        default=None, description="Filter to one service name. None = all services."
    )
    window: Optional[TimeWindow] = Field(
        default=None,
        description=(
            "Optional additional narrowing within the case's allowed "
            "diagnostic window. Cannot extend beyond it — the adapter "
            "intersects this with the diagnostic window, never overrides it."
        ),
    )
    max_rows: Optional[int] = Field(
        default=None,
        gt=0,
        description="Bounded sampling cap for development / prompt-building use.",
    )


class ModalityAvailability(BaseModel):
    """Which telemetry modalities exist for a given case (RE1 has no logs/traces)."""

    model_config = ConfigDict(extra="forbid")

    metrics: bool = False
    logs: bool = False
    traces: bool = False

    def as_set(self) -> set:
        return {name for name, present in self.model_dump().items() if present}


class TelemetrySummary(BaseModel):
    """
    Bounded, human/LLM-readable summary of one case's telemetry — this,
    not raw tables, is what crosses into a prompt or an API response body.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    modalities: ModalityAvailability
    service_names: List[str] = Field(default_factory=list)
    row_counts: Dict[Modality, int] = Field(
        default_factory=dict,
        description="Row count per modality actually present, after any bounding applied.",
    )
    time_range: TimeWindow = Field(
        description="Actual [start, end] covered by the returned/sampled rows."
    )
    sample: Dict[Modality, list] = Field(
        default_factory=dict,
        description=(
            "Small bounded sample of rows per modality (e.g. first N), for "
            "quick inspection. Not the full dataset — size capped by the "
            "caller's TelemetryQuery.max_rows or a sane default in the adapter."
        ),
    )