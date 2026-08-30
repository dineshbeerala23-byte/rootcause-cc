from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.schemas.claims import ClaimVerdict, ClaimVerificationResult
from app.schemas.evidence import EvidenceBundle, EvidenceItem
from app.schemas.graph import ServiceGraph
from app.schemas.incident import CaseMetadata, DiagnosticWindow, NormalizedIncident
from app.schemas.reasoning import CandidateCause, RCACandidateResult
from app.services.final_result.service import FinalResultAssemblyService
from app.services.llm.base import LLMProvider

AS_OF = datetime(2026, 1, 1, tzinfo=timezone.utc)
INJECT_TIME = datetime(2025, 12, 31, tzinfo=timezone.utc)


class FakeAdapter:
    """Fake DatasetAdapter -- only implements get_incident, per this
    milestone's scope. Records the case_id it was called with so tests
    can assert on it."""

    def __init__(self, incident: NormalizedIncident):
        self._incident = incident
        self.last_case_id = None

    def get_incident(self, case_id: str) -> NormalizedIncident:
        self.last_case_id = case_id
        return self._incident


class FakeEvidenceService:
    def __init__(self, bundle: EvidenceBundle):
        self._bundle = bundle
        self.last_case_id = None
        self.last_query = None

    def get_evidence(self, case_id: str, query) -> EvidenceBundle:
        self.last_case_id = case_id
        self.last_query = query
        return self._bundle


class FakeTopologyProvider:
    def __init__(self, graph: ServiceGraph):
        self._graph = graph
        self.last_system = None

    def get_graph(self, system) -> ServiceGraph:
        self.last_system = system
        return self._graph


class FakeReasoningService:
    def __init__(self, candidate_result: RCACandidateResult):
        self._candidate_result = candidate_result
        self.last_mode = None

    async def analyze(self, incident, mode="deterministic", evidence_query=None):
        self.last_mode = mode
        return self._candidate_result


class FakeClaimsService:
    def __init__(self, verification_result: ClaimVerificationResult):
        self._verification_result = verification_result

    async def analyze(self, candidate_result, evidence, graph):
        return self._verification_result


class FakeLLMProvider(LLMProvider):
    def __init__(self, name: str = "fake-provider", model: str = "fake-model"):
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return ""


@pytest.fixture
def incident() -> NormalizedIncident:
    return NormalizedIncident(
        metadata=CaseMetadata(
            case_id="case-0001",
            dataset_suite="RE1",
            system="online-boutique",
            available_modalities={"metrics": True, "logs": False, "traces": False},
        ),
        diagnostic_window=DiagnosticWindow(inject_time=INJECT_TIME, as_of=AS_OF),
    )


@pytest.fixture
def evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        case_id="case-0001",
        as_of=AS_OF,
        modalities_used=["metrics"],
        items=[
            EvidenceItem(
                service="cartservice",
                metric_type="cpu",
                column_name="cartservice_cpu",
                summary={"mean": 0.9},
                score=0.8,
            ),
        ],
    )


@pytest.fixture
def service_graph() -> ServiceGraph:
    return ServiceGraph(system="online-boutique", nodes=[], edges=[], notes=[])


@pytest.fixture
def candidate_result() -> RCACandidateResult:
    return RCACandidateResult(
        case_id="case-0001",
        as_of=AS_OF,
        method="deterministic",
        candidates=[
            CandidateCause(
                service="cartservice",
                score=0.8,
                supporting_evidence=["cartservice_cpu"],
                graph_support=None,
            ),
        ],
        notes=[],
    )


@pytest.fixture
def verification_result() -> ClaimVerificationResult:
    return ClaimVerificationResult(
        case_id="case-0001",
        as_of=AS_OF,
        verdicts=[
            ClaimVerdict(
                claim_id="c1",
                candidate_service="cartservice",
                status="supported",
                matched_evidence=["cartservice_cpu"],
                unmatched_evidence=[],
            )
        ],
        notes=[],
    )


@pytest.fixture
def final_result_service() -> FinalResultAssemblyService:
    return FinalResultAssemblyService()


@pytest.fixture
def pipeline_service_factory(
    incident, evidence_bundle, service_graph, candidate_result, verification_result, final_result_service
):
    """
    Returns a zero-arg factory building an RCAPipelineService wired to
    fresh fake collaborators, plus the fakes themselves (as a dict) so
    tests can assert on what each collaborator was called with, without
    needing to import the Fake* classes across a module boundary.
    """

    def _build():
        from app.services.pipeline.service import RCAPipelineService

        adapter = FakeAdapter(incident)
        evidence_service = FakeEvidenceService(evidence_bundle)
        topology_provider = FakeTopologyProvider(service_graph)
        reasoning_service = FakeReasoningService(candidate_result)
        claims_service = FakeClaimsService(verification_result)
        llm_provider = FakeLLMProvider()

        service = RCAPipelineService(
            adapter=adapter,
            evidence_service=evidence_service,
            topology_provider=topology_provider,
            reasoning_service=reasoning_service,
            claims_service=claims_service,
            final_result_service=final_result_service,
            llm_provider=llm_provider,
        )
        fakes = {
            "adapter": adapter,
            "evidence_service": evidence_service,
            "topology_provider": topology_provider,
            "reasoning_service": reasoning_service,
            "claims_service": claims_service,
            "llm_provider": llm_provider,
        }
        return service, fakes

    return _build
