# ============================================================
# app/services/reasoning/deterministic.py  (NEW)
# ============================================================
"""
Rule-based RCA candidate ranking. This is the authoritative ranking in
this milestone -- llm_reasoner.py may annotate its output with rationale
text but must never reorder, add, or remove candidates it produces.

Consumes only EvidenceBundle + ServiceGraph + NormalizedIncident. No
telemetry access, no LLM, no ground truth. Deterministic and fully
testable offline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.schemas.evidence import EvidenceBundle
from app.schemas.graph import ServiceGraph
from app.schemas.incident import NormalizedIncident
from app.schemas.reasoning import CandidateCause, RCACandidateResult
from app.services.reasoning.base import CandidateGenerator

_DEFAULT_MAX_CANDIDATES = 5


class DeterministicCandidateGenerator(CandidateGenerator):
    """
    Ranking: sum of per-item deviation scores (EvidenceItem.score), grouped
    by EvidenceItem.service. Items with service=None are not attributable
    to a candidate and are ignored for ranking purposes.

    Graph adjacency is used only to annotate each candidate with its direct
    neighbors in the static topology -- never to invent an edge ServiceGraph
    does not contain, and never populated at all when the graph for this
    system has zero edges (e.g. train-ticket, where edge research was not
    performed).
    """

    def __init__(self, max_candidates: int = _DEFAULT_MAX_CANDIDATES):
        self._max_candidates = max_candidates

    def generate(
        self,
        incident: NormalizedIncident,
        evidence: EvidenceBundle,
        graph: ServiceGraph,
    ) -> RCACandidateResult:
        notes: List[str] = []

        per_service_score: Dict[str, float] = defaultdict(float)
        per_service_evidence: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

        for item in evidence.items:
            if item.service is None:
                continue
            per_service_score[item.service] += item.score
            per_service_evidence[item.service].append((item.score, item.column_name))

        if not per_service_score:
            notes.append(
                "No service-attributed evidence items available; no candidates generated."
            )
            return RCACandidateResult(
                case_id=incident.case_id,
                as_of=evidence.as_of,
                method="deterministic",
                candidates=[],
                notes=notes,
            )

        graph_has_edges = len(graph.edges) > 0
        if not graph_has_edges:
            notes.append(
                f"Service graph for system '{graph.system}' has no edges; "
                "graph_support omitted for all candidates."
            )

        adjacency: Dict[str, List[str]] = defaultdict(list)
        if graph_has_edges:
            for edge in graph.edges:
                adjacency[edge.source].append(edge.target)
                adjacency[edge.target].append(edge.source)

        ranked_services = sorted(
            per_service_score.keys(), key=lambda s: (-per_service_score[s], s)
        )

        candidates: List[CandidateCause] = []
        for service in ranked_services[: self._max_candidates]:
            evidence_sorted = sorted(per_service_evidence[service], key=lambda t: -t[0])
            supporting_evidence = [column for _, column in evidence_sorted]

            graph_support: Optional[List[str]] = None
            if graph_has_edges:
                graph_support = sorted(set(adjacency.get(service, [])))

            candidates.append(
                CandidateCause(
                    service=service,
                    score=per_service_score[service],
                    supporting_evidence=supporting_evidence,
                    graph_support=graph_support,
                )
            )

        return RCACandidateResult(
            case_id=incident.case_id,
            as_of=evidence.as_of,
            method="deterministic",
            candidates=candidates,
            notes=notes,
        )