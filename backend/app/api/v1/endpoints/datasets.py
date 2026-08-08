"""
Safe RCAEval dataset endpoints (Phase 3, Part 7).

Thin by design, same convention as extraction.py: request/response
shaping only. All dataset logic lives in app.services.datasets.*
(discovery, raw_loader, adapter). Errors raised there are RootCauseError
subclasses, handled by the global exception handlers in app.main —
nothing here needs its own try/except.

Ground truth (app.schemas.ground_truth.GroundTruth) is never imported or
returned here. There is no ?include_ground_truth parameter and none
should be added — ground truth stays reachable only from evaluation-only
code, per the Phase 3 design.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import get_rcaeval_adapter
from app.schemas.incident import CaseMetadata, NormalizedIncident
from app.schemas.telemetry import TelemetryQuery, TelemetrySummary
from app.services.datasets.base import DatasetAdapter

logger = logging.getLogger(__name__)

router = APIRouter()


class TelemetryRowsResponse(BaseModel):
    """
    Bounded telemetry rows for one case, already filtered to the
    diagnostic cutoff and any service/max_rows constraints from the
    query. This, not raw pandas data, is what the API returns.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    traces: List[Dict[str, Any]] = Field(default_factory=list)


def _dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    JSON-safe conversion of a (possibly numpy-typed) DataFrame to plain
    Python records, via pandas' own JSON encoder rather than
    DataFrame.to_dict() (which can leave numpy scalar types that the
    default JSON encoder can't serialize).
    """
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


@router.get(
    "",
    response_model=List[CaseMetadata],
    summary="List available RCAEval cases",
    description=(
        "Returns safe, answer-free metadata for available cases, identified "
        "only by opaque case_id (e.g. 'case-0001') — never the raw RCAEval "
        "directory name."
    ),
)
async def list_cases(
    adapter: DatasetAdapter = Depends(get_rcaeval_adapter),
) -> List[CaseMetadata]:
    return adapter.list_cases()


@router.get(
    "/{case_id}",
    response_model=NormalizedIncident,
    summary="Inspect a normalized case",
    description="Returns the ground-truth-free normalized incident for one opaque case_id.",
)
async def get_case(
    case_id: str,
    adapter: DatasetAdapter = Depends(get_rcaeval_adapter),
) -> NormalizedIncident:
    return adapter.get_incident(case_id)


@router.get(
    "/{case_id}/telemetry/summary",
    response_model=TelemetrySummary,
    summary="Bounded telemetry summary for a case",
    description=(
        "Returns a bounded summary of available telemetry (modalities, row "
        "counts, small sample), filtered to the case's diagnostic cutoff. "
        "Never the full raw dataset."
    ),
)
async def get_case_telemetry_summary(
    case_id: str,
    adapter: DatasetAdapter = Depends(get_rcaeval_adapter),
) -> TelemetrySummary:
    incident = adapter.get_incident(case_id)
    return adapter.get_telemetry_summary(case_id, incident.diagnostic_window)


@router.get(
    "/{case_id}/telemetry",
    response_model=TelemetryRowsResponse,
    summary="Bounded telemetry rows for a case",
    description=(
        "Returns telemetry rows filtered to the case's diagnostic cutoff, "
        "optionally narrowed by service and capped by max_rows."
    ),
)
async def get_case_telemetry(
    case_id: str,
    service: Optional[str] = Query(default=None, description="Filter to one service name."),
    max_rows: int = Query(default=20, gt=0, description="Bounded row cap per modality."),
    adapter: DatasetAdapter = Depends(get_rcaeval_adapter),
) -> TelemetryRowsResponse:
    query = TelemetryQuery(service=service, max_rows=max_rows)
    handle = adapter.get_telemetry(case_id, query)

    return TelemetryRowsResponse(
        case_id=case_id,
        metrics=_dataframe_to_records(handle.metrics()),
        logs=_dataframe_to_records(handle.logs()),
        traces=_dataframe_to_records(handle.traces()),
    )