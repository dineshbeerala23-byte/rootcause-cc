from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.incident import System

ComponentKind = str  # "application" | "database" | "infrastructure" -- see _KNOWN_KINDS below
_KNOWN_KINDS = {"application", "database", "infrastructure"}


class ServiceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    system: System
    kind: str  # one of _KNOWN_KINDS; validated at construction via ServiceGraph, not per-node


class ServiceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str


class ServiceGraph(BaseModel):
    """
    Per-system static topology. `notes` carries forward unresolved
    research findings as structured data (e.g. an ambiguous node's
    identity, or "edges not yet researched") rather than a code
    comment nobody reads at the API/consumer boundary.
    """

    model_config = ConfigDict(extra="forbid")
    system: System
    nodes: List[ServiceNode] = Field(default_factory=list)
    edges: List[ServiceEdge] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)