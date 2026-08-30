from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

from app.schemas.telemetry import TimeWindow


class EvidenceQuery(BaseModel):
    services: Optional[List[str]] = None
    metric_types: Optional[List[str]] = None
    window: Optional[TimeWindow] = None
    max_items: int = 20


class EvidenceItem(BaseModel):
    service: Optional[str]
    metric_type: Optional[str]
    column_name: str
    summary: Dict[str, float]
    score: float


class EvidenceBundle(BaseModel):
    case_id: str
    as_of: datetime
    modalities_used: List[str]
    items: List[EvidenceItem]