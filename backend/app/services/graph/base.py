from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.graph import ServiceGraph
from app.schemas.incident import System


class TopologyProvider(ABC):
    """Dataset-agnostic interface, mirrors DatasetAdapter's pattern."""

    @abstractmethod
    def get_graph(self, system: System) -> ServiceGraph:
        raise NotImplementedError