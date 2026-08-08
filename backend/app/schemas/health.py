"""
Pydantic schemas for the /health endpoint.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    """Status of a single downstream dependency the platform relies on."""

    name: str
    status: Literal["ok", "not_configured", "unavailable"]
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_name: str
    environment: str
    version: str
    timestamp: datetime
    components: List[ComponentStatus] = Field(default_factory=list)