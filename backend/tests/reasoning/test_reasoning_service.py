import json
from pathlib import Path

import pytest

from app.schemas.evidence import EvidenceBundle, EvidenceQuery
from app.schemas.graph import ServiceGraph
from app.services.llm.base import LLMProvider
from app.services.reasoning.llm_reasoner import LLMRationaleAnnotator
from app.services.reasoning.service import RCAReasoningService

from .conftest import make_evidence, make_graph_with_edges, make_graph_without_edges, make_incident


class FakeEvidenceService:
    """Duck-typed stand-in for EvidenceRetrievalService -- never touches DatasetAdapter."""

    def __init__(self, bundle: EvidenceBundle):
        self._bundle = bundle
        self.last_query: EvidenceQuery | None = None

    def get_evidence(self, case_id: str, query: EvidenceQuery) -> EvidenceBundle:
        self.last_query = query
        return self._bundle


class FakeTopologyProvider:
    """Duck-typed stand-in for TopologyProvider."""

    def __init__(self, graph: ServiceGraph):
        self._graph = graph

    def get_graph(self, system: str) -> ServiceGraph:
        return self._graph


class FakeLLMProvider(LLMProvider):
    def __init__(self, response: str):
        self._response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


@pytest.mark.asyncio
async def test_deterministic_mode_requires_no_llm_provider(incident):
    evidence_service = FakeEvidenceService(make_evidence())
    topology_provider = FakeTopologyProvider(make_graph_with_edges())
    service = RCAReasoningService(evidence_service, topology_provider)

    result = await service.analyze(incident, mode="deterministic")

    assert result.method == "deterministic"
    assert len(result.candidates) > 0


@pytest.mark.asyncio
async def test_as_of_matches_evidence_bundle_exactly(incident):
    evidence = make_evidence()
    evidence_service = FakeEvidenceService(evidence)
    topology_provider = FakeTopologyProvider(make_graph_with_edges())
    service = RCAReasoningService(evidence_service, topology_provider)

    result = await service.analyze(incident)

    assert result.as_of == evidence.as_of


@pytest.mark.asyncio
async def test_llm_assisted_mode_without_annotator_falls_back_with_note(incident):
    evidence_service = FakeEvidenceService(make_evidence())
    topology_provider = FakeTopologyProvider(make_graph_with_edges())
    service = RCAReasoningService(evidence_service, topology_provider)

    result = await service.analyze(incident, mode="llm-assisted")

    assert result.method == "deterministic"
    assert any("no rationale annotator" in note.lower() for note in result.notes)


@pytest.mark.asyncio
async def test_llm_assisted_mode_with_annotator_adds_rationale_only(incident):
    evidence_service = FakeEvidenceService(make_evidence())
    topology_provider = FakeTopologyProvider(make_graph_with_edges())
    response = json.dumps({"rationales": {"cartservice": "top mover", "redis": "downstream dependency"}})
    annotator = LLMRationaleAnnotator(FakeLLMProvider(response))
    service = RCAReasoningService(evidence_service, topology_provider, rationale_annotator=annotator)

    deterministic_only = await RCAReasoningService(evidence_service, topology_provider).analyze(incident)
    result = await service.analyze(incident, mode="llm-assisted")

    assert result.method == "llm-assisted"
    assert [c.service for c in result.candidates] == [c.service for c in deterministic_only.candidates]
    assert [c.score for c in result.candidates] == [c.score for c in deterministic_only.candidates]
    assert result.candidates[0].rationale is not None


@pytest.mark.asyncio
async def test_invalid_mode_raises_value_error(incident):
    evidence_service = FakeEvidenceService(make_evidence())
    topology_provider = FakeTopologyProvider(make_graph_with_edges())
    service = RCAReasoningService(evidence_service, topology_provider)

    with pytest.raises(ValueError):
        await service.analyze(incident, mode="not-a-real-mode")


@pytest.mark.asyncio
async def test_evidence_service_never_receives_a_dataset_adapter_reference(incident):
    """
    RCAReasoningService only ever calls evidence_service.get_evidence and
    topology_provider.get_graph -- confirms no other method (e.g. one that
    could reach get_ground_truth) is ever invoked on its collaborators.
    """
    evidence_service = FakeEvidenceService(make_evidence())
    topology_provider = FakeTopologyProvider(make_graph_with_edges())
    service = RCAReasoningService(evidence_service, topology_provider)

    await service.analyze(incident)

    assert evidence_service.last_query is not None


REASONING_SOURCE_FILES = [
    "app/schemas/reasoning.py",
    "app/services/reasoning/base.py",
    "app/services/reasoning/deterministic.py",
    "app/services/reasoning/llm_reasoner.py",
    "app/services/reasoning/service.py",
]


def test_reasoning_source_never_imports_or_uses_ground_truth():
    """
    Guardrail: scans the actual reasoning source files for any IMPORT of, or
    constructed/annotated USE of, GroundTruth -- so the isolation boundary is
    enforced by a test rather than by reviewer discipline alone. Explanatory
    prose in docstrings/comments that merely *names* ground_truth.py (to
    document that it is deliberately excluded) is not itself a violation and
    is not flagged.
    """
    repo_root = Path(__file__).resolve().parents[2]
    forbidden_patterns = [
        "from app.schemas.ground_truth",
        "from app.schemas import ground_truth",
        "import app.schemas.ground_truth",
        "GroundTruth(",
        ": GroundTruth",
        "-> GroundTruth",
        "DatasetAdapter",
        "get_ground_truth(",
    ]
    for relative_path in REASONING_SOURCE_FILES:
        source = (repo_root / relative_path).read_text()
        for pattern in forbidden_patterns:
            assert pattern not in source, f"{relative_path} contains forbidden pattern: {pattern!r}"