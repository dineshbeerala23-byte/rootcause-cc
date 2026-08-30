# ============================================================
# app/schemas/pipeline.py  (NEW)
# ============================================================
"""
Request/response schemas for the end-to-end RCA pipeline surface
(API endpoint + CLI command).

Thin wrapper around FinalRCAResult -- same envelope convention as
app/schemas/extraction.py's ExtractionResponse (provider/model + result).
Carries no ground-truth fields; case_id is always the opaque id, passed
straight through from FinalRCAResult.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.schemas.final_result import FinalRCAResult
from app.schemas.reasoning import ReasoningMethod


class RCAAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ReasoningMethod = "deterministic"


class RCAAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    mode: ReasoningMethod
    provider: str
    model: str
    result: FinalRCAResult
