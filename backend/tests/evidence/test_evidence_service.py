from datetime import datetime, timezone

import pandas as pd
import pytest

from app.schemas.evidence import EvidenceQuery
from app.schemas.incident import CaseMetadata, DiagnosticWindow, NormalizedIncident
from app.schemas.telemetry import ModalityAvailability, TimeWindow
from app.services.evidence.service import EvidenceRetrievalService


class _FakeTelemetryHandle:
    def __init__(self, metrics_df: pd.DataFrame):
        self._metrics_df = metrics_df

    def metrics(self) -> pd.DataFrame:
        return self._metrics_df

    def logs(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["time"])

    def traces(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["time"])


class _FakeAdapter:
    def __init__(self):
        n = 40
        self._times = list(range(1_700_000_000, 1_700_000_000 + n))
        self._full_df = pd.DataFrame(
            {
                "time": self._times,
                "adservice_cpu": [10.0] * n,
                "cartservice_latency": [5.0] * (n // 2) + [50.0] * (n // 2),
                "frontend_error": [0.0] * n,
                "weird_unmapped_col": [1.0] * n,
            }
        )

    def get_incident(self, case_id: str) -> NormalizedIncident:
        metadata = CaseMetadata(
            case_id=case_id,
            dataset_suite="RE1",
            system="online-boutique",
            available_modalities=ModalityAvailability(metrics=True, logs=False, traces=False),
        )
        window = DiagnosticWindow(
            inject_time=datetime.fromtimestamp(self._times[0], tz=timezone.utc),
            as_of=None,
        )
        return NormalizedIncident(metadata=metadata, diagnostic_window=window)

    def get_telemetry(self, case_id: str, query):
        df = self._full_df
        if query.window is not None and query.window.end is not None:
            cutoff = query.window.end.timestamp()
            df = df[df["time"] <= cutoff]
        return _FakeTelemetryHandle(df)

    def list_cases(self):
        raise NotImplementedError

    def get_telemetry_summary(self, case_id, window):
        raise NotImplementedError

    def get_ground_truth(self, case_id):
        raise NotImplementedError


@pytest.fixture
def fake_adapter():
    return _FakeAdapter()


@pytest.fixture
def service(fake_adapter):
    return EvidenceRetrievalService(fake_adapter)


def test_bounded_by_max_items(service):
    bundle = service.get_evidence("case-0001", EvidenceQuery(max_items=2))
    assert len(bundle.items) == 2


def test_deterministic_ordering(service):
    bundle1 = service.get_evidence("case-0001", EvidenceQuery(max_items=10))
    bundle2 = service.get_evidence("case-0001", EvidenceQuery(max_items=10))
    assert [item.column_name for item in bundle1.items] == [
        item.column_name for item in bundle2.items
    ]


def test_service_filter(service):
    bundle = service.get_evidence(
        "case-0001", EvidenceQuery(services=["cartservice"], max_items=10)
    )
    assert len(bundle.items) == 1
    assert bundle.items[0].service == "cartservice"


def test_metric_type_filter(service):
    bundle = service.get_evidence(
        "case-0001", EvidenceQuery(metric_types=["cpu"], max_items=10)
    )
    assert len(bundle.items) == 1
    assert bundle.items[0].metric_type == "cpu"


def test_unparsed_column_kept_not_dropped(service):
    bundle = service.get_evidence("case-0001", EvidenceQuery(max_items=10))
    column_names = [item.column_name for item in bundle.items]
    assert "weird_unmapped_col" in column_names


def test_cutoff_respected(fake_adapter):
    service = EvidenceRetrievalService(fake_adapter)
    midpoint = datetime.fromtimestamp(fake_adapter._times[19], tz=timezone.utc)
    bundle = service.get_evidence(
        "case-0001",
        EvidenceQuery(services=["cartservice"], window=TimeWindow(end=midpoint), max_items=10),
    )
    assert bundle.items[0].summary["max"] == 5.0


def test_no_ground_truth_leakage(service):
    bundle = service.get_evidence("case-0001", EvidenceQuery(max_items=10))
    serialized = bundle.model_dump_json()
    for forbidden in ("root_cause_service", "fault_type", "root_cause_indicator"):
        assert forbidden not in serialized


def test_case_id_passthrough(service):
    bundle = service.get_evidence("case-0042", EvidenceQuery(max_items=1))
    assert bundle.case_id == "case-0042"