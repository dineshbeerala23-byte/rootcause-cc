from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ClaimGenerationMethod = Literal["llm-generated"]
ClaimVerificationStatus = Literal["supported", "unsupported", "uncertain"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    candidate_service: str
    statement: str
    cited_evidence: List[str] = Field(default_factory=list)


class ClaimGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    as_of: datetime
    method: ClaimGenerationMethod
    claims: List[Claim] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    candidate_service: str
    status: ClaimVerificationStatus
    matched_evidence: List[str] = Field(default_factory=list)
    unmatched_evidence: List[str] = Field(default_factory=list)


class ClaimVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    as_of: datetime
    verdicts: List[ClaimVerdict] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)