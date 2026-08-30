from __future__ import annotations

from app.schemas.graph import ServiceGraph
from app.schemas.incident import System
from app.services.graph.base import TopologyProvider
from app.services.graph.static_data import _GRAPHS_BY_SYSTEM


class StaticTopologyProvider(TopologyProvider):
    """v1: returns hard-coded ServiceGraph data. No Neo4j, no dynamic discovery."""

    def get_graph(self, system: System) -> ServiceGraph:
        return _GRAPHS_BY_SYSTEM[system]