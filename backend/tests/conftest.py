"""
Shared pytest fixtures.

`FakeLLMProvider` is a test double for app.services.llm.base.LLMProvider.
Tests use it (via the `client` fixture below) to override
app.api.deps.get_llm_provider, so the extraction endpoint's tests never
make a real network call to Groq — they only exercise our own service and
parsing logic against a canned response.

The RCAEval fixtures below (rcaeval_fixture_dir, rcaeval_settings,
rcaeval_adapter, dataset_client) do the equivalent for Phase 3 dataset
tests: everything points at tests/fixtures/rcaeval_mini/, a tiny synthetic
directory tree, so no test ever downloads real RCAEval data or touches
the internet.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_llm_provider, get_rcaeval_adapter
from app.core.config import Settings
from app.main import app
from app.services.datasets.rcaeval.adapter import RCAEvalAdapter
from app.services.datasets.rcaeval.discovery import RCAEvalDiscovery
from app.services.llm.base import LLMProvider


class FakeLLMProvider(LLMProvider):
    """In-memory stand-in for a real LLM provider. No network calls."""

    def __init__(self, response: str = "", raise_error: Optional[Exception] = None) -> None:
        self.response = response
        self.raise_error = raise_error
        self.calls: List[Tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.raise_error is not None:
            raise self.raise_error
        return self.response


@pytest.fixture
def fake_llm_provider() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def client(fake_llm_provider: FakeLLMProvider) -> TestClient:
    """TestClient with the real Groq provider swapped for FakeLLMProvider."""
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm_provider
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def rcaeval_fixture_dir() -> Path:
    """Path to the tiny synthetic RCAEval directory tree used by all Phase 3 tests."""
    return Path(__file__).parent / "fixtures" / "rcaeval_mini"


@pytest.fixture
def rcaeval_settings(rcaeval_fixture_dir: Path) -> Settings:
    """Settings with RCAEVAL_DATA_DIR pointed at the synthetic fixture dir."""
    return Settings(RCAEVAL_DATA_DIR=str(rcaeval_fixture_dir))


@pytest.fixture
def rcaeval_adapter(rcaeval_settings: Settings) -> RCAEvalAdapter:
    return RCAEvalAdapter(discovery=RCAEvalDiscovery(settings=rcaeval_settings))


@pytest.fixture
def dataset_client(fake_llm_provider: FakeLLMProvider, rcaeval_adapter: RCAEvalAdapter) -> TestClient:
    """
    TestClient with both the LLM provider and the RCAEval adapter swapped
    for fixture-backed test doubles, so dataset endpoint tests never touch
    real Groq or a real RCAEval download.
    """
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm_provider
    app.dependency_overrides[get_rcaeval_adapter] = lambda: rcaeval_adapter
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()