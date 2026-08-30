from datetime import datetime, timezone

import pytest

from app.schemas.evidence import EvidenceBundle, EvidenceItem
from app.schemas.graph import ServiceEdge, ServiceGraph, ServiceNode
from app.schemas.incident import CaseMetadata, DiagnosticWindow, NormalizedIncident
from app.schemas.telemetry import ModalityAvailability

INJECT_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
AS_OF = datetime(2025, 1, 1, 0, 10, tzinfo=timezone.utc)


def make_incident(case_id: str = "case-0001", system: str = "online-boutique") -> NormalizedIncident:
    return NormalizedIncident(
        metadata=CaseMetadata(
            case_id=case_id,
            dataset_suite="RE1",
            system=system,
            available_modalities=ModalityAvailability(metrics=True, logs=False, traces=False),
        ),
        diagnostic_window=DiagnosticWindow(inject_time=INJECT_TIME, as_of=AS_OF),
    )


def make_evidence(case_id: str = "case-0001") -> EvidenceBundle:
    return EvidenceBundle(
        case_id=case_id,
        as_of=AS_OF,
        modalities_used=["metrics"],
        items=[
            EvidenceItem(
                service="cartservice",
                metric_type="cpu",
                column_name="cartservice_cpu",
                summary={"mean": 1.0, "std": 0.1, "min": 0.5, "max": 2.0, "last": 1.9},
                score=9.5,
            ),
            EvidenceItem(
                service="cartservice",
                metric_type="mem",
                column_name="cartservice_mem",
                summary={"mean": 1.0, "std": 0.1, "min": 0.5, "max": 2.0, "last": 1.9},
                score=1.0,
            ),
            EvidenceItem(
                service="redis",
                metric_type="cpu",
                column_name="redis_cpu",
                summary={"mean": 1.0, "std": 0.1, "min": 0.5, "max": 2.0, "last": 1.9},
                score=4.0,
            ),
            EvidenceItem(
                service=None,
                metric_type="cpu",
                column_name="unattributed_cpu",
                summary={"mean": 1.0, "std": 0.1, "min": 0.5, "max": 2.0, "last": 1.9},
                score=99.0,
            ),
        ],
    )


def make_graph_with_edges(system: str = "online-boutique") -> ServiceGraph:
    return ServiceGraph(
        system=system,
        nodes=[
            ServiceNode(name="cartservice", system=system, kind="application"),
            ServiceNode(name="redis", system=system, kind="database"),
            ServiceNode(name="checkoutservice", system=system, kind="application"),
        ],
        edges=[
            ServiceEdge(source="cartservice", target="redis"),
            ServiceEdge(source="checkoutservice", target="cartservice"),
        ],
        notes=[],
    )


def make_graph_without_edges(system: str = "train-ticket") -> ServiceGraph:
    return ServiceGraph(
        system=system,
        nodes=[ServiceNode(name="ts-auth-service", system=system, kind="application")],
        edges=[],
        notes=["Edge research NOT performed -- edges intentionally empty."],
    )


@pytest.fixture
def incident() -> NormalizedIncident:
    return make_incident()


@pytest.fixture
def evidence() -> EvidenceBundle:
    return make_evidence()


@pytest.fixture
def graph_with_edges() -> ServiceGraph:
    return make_graph_with_edges()


@pytest.fixture
def graph_without_edges() -> ServiceGraph:
    return make_graph_without_edges()