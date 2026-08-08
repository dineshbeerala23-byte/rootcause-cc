# ============================================================
# app/schemas/incident.py  (NEW)
# ============================================================
"""
Normalized, dataset-agnostic incident representation.

NormalizedIncident is the ONLY incident-shaped object that should ever
reach inference code, a prompt, or a default API response body. It
carries no ground-truth fields — see app/schemas/ground_truth.py for the
deliberately separate GroundTruth model, joined only via opaque case_id.

case_id here is always the OPAQUE id (e.g. "case-0001"), never the raw
RCAEval directory name — the real directory name lives only in
app/services/datasets/rcaeval/case_id_registry.py (Part 3), reachable
only from the discovery/evaluation layer.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from datetime import datetime

from app.schemas.telemetry import ModalityAvailability

DatasetSuite = Literal["RE1", "RE2", "RE3"]
System = Literal["online-boutique", "sock-shop", "train-ticket"]


class DiagnosticWindow(BaseModel):
    """
    The time boundary an RCA method is allowed to see for this case.

    inject_time: when RCAEval injected the fault (from inject_time.txt).
    Not itself ground truth in the "which service" sense, but treated
    carefully anyway — see Part 5 notes on whether/how it's exposed.

    as_of: the diagnostic cutoff. If None, the adapter (Part 5) MUST
    resolve it to inject_time + a small fixed buffer before any telemetry
    is read — never silently to "all data in the file". This field is
    intentionally Optional here so unresolved and resolved states are
    distinguishable; by the time a NormalizedIncident is actually handed
    to inference code, as_of must be set (the adapter enforces this, not
    this schema — Pydantic alone can't express "required after some
    processing step").
    """

    model_config = ConfigDict(extra="forbid")

    inject_time: datetime
    as_of: Optional[datetime] = None

    @model_validator(mode="after")
    def _as_of_not_before_inject(self) -> "DiagnosticWindow":
        if self.as_of is not None and self.as_of < self.inject_time:
            raise ValueError("as_of must not be before inject_time")
        return self


class CaseMetadata(BaseModel):
    """Safe, answer-free descriptive metadata about one case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., description="Opaque case ID, e.g. 'case-0001'.")
    dataset_suite: DatasetSuite
    system: System
    available_modalities: ModalityAvailability


class NormalizedIncident(BaseModel):
    """
    The dataset-agnostic incident representation. Any current or future
    RCA method, verification module, or prompt-builder should depend on
    THIS type, never on RCAEval's raw file layout directly.
    """

    model_config = ConfigDict(extra="forbid")

    metadata: CaseMetadata
    diagnostic_window: DiagnosticWindow

    @property
    def case_id(self) -> str:
        """Convenience passthrough — most callers just want the id."""
        return self.metadata.case_id