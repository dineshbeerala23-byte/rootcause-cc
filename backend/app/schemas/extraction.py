"""
Pydantic schemas for Phase 2 — LLM-based entity and relationship extraction.

These are the structured, validated shapes that sit between raw LLM output
and the API response. Nothing in app/api or app/services should ever pass
unvalidated LLM text back to a client — everything is parsed into these
models first (see app.services.extraction_service).
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """
    Entity categories relevant to incident/root-cause analysis. This is the
    vocabulary Phase 3's knowledge graph will use as node labels, so it is
    kept deliberately small and infrastructure/incident-oriented rather
    than an open-ended free-text type.
    """

    SERVICE = "Service"
    APPLICATION = "Application"
    HOST = "Host"
    CONTAINER = "Container"
    POD = "Pod"
    DATABASE = "Database"
    DEPLOYMENT = "Deployment"
    ERROR = "Error"
    INCIDENT = "Incident"
    ALERT = "Alert"
    DEPENDENCY = "Dependency"
    CONFIGURATION = "Configuration"
    CLOUD_RESOURCE = "CloudResource"


class RelationshipType(str, Enum):
    """
    Relationship categories connecting extracted entities. Mirrors the edge
    labels Phase 3 will persist in the knowledge graph.
    """

    DEPENDS_ON = "DEPENDS_ON"
    CALLS = "CALLS"
    RUNS_ON = "RUNS_ON"
    CONNECTED_TO = "CONNECTED_TO"
    CAUSED_BY = "CAUSED_BY"
    AFFECTS = "AFFECTS"
    TRIGGERED = "TRIGGERED"
    FAILED_DUE_TO = "FAILED_DUE_TO"
    PART_OF = "PART_OF"


class Entity(BaseModel):
    """A single extracted entity."""

    id: str = Field(
        ...,
        min_length=1,
        description="Short identifier unique within this extraction (e.g. 'e1'). "
        "Used by relationships to reference this entity.",
    )
    name: str = Field(..., min_length=1, description="Human-readable name of the entity.")
    type: EntityType
    description: Optional[str] = Field(default=None, description="Brief context pulled from the source text.")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Optional extra key/value details.")


class Relationship(BaseModel):
    """A directed relationship between two extracted entities."""

    source_id: str = Field(..., min_length=1, description="id of the entity this relationship starts from.")
    target_id: str = Field(..., min_length=1, description="id of the entity this relationship points to.")
    type: RelationshipType
    description: Optional[str] = Field(default=None, description="Brief context pulled from the source text.")


class ExtractionResult(BaseModel):
    """Structured result of an extraction run, ready for Phase 3 to consume."""

    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered while parsing the LLM output "
        "(e.g. an invalid type or a dangling relationship reference that was dropped).",
    )


class ExtractionRequest(BaseModel):
    """Request body for POST /api/v1/extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Raw cloud/system/incident text to analyze.",
    )


class ExtractionResponse(BaseModel):
    """Response body for POST /api/v1/extraction."""

    provider: str = Field(..., description="Name of the LLM provider that produced this result.")
    model: str = Field(..., description="Model identifier used for this extraction.")
    result: ExtractionResult