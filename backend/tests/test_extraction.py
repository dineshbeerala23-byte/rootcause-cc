"""
Tests for the Phase 2 extraction endpoint.

All tests use the `client` / `fake_llm_provider` fixtures from conftest.py,
so nothing here makes a real Groq API call.
"""

import json

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMProviderError
from app.main import app

VALID_LLM_RESPONSE = json.dumps(
    {
        "entities": [
            {"id": "e1", "name": "checkout-service", "type": "Service"},
            {"id": "e2", "name": "orders-db", "type": "Database"},
        ],
        "relationships": [
            {"source_id": "e1", "target_id": "e2", "type": "DEPENDS_ON"},
        ],
    }
)


def test_extraction_success(client, fake_llm_provider):
    fake_llm_provider.response = VALID_LLM_RESPONSE

    response = client.post(
        "/api/v1/extraction",
        json={"text": "The checkout-service depends on the orders-db database."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "fake"
    assert body["model"] == "fake-model"
    assert len(body["result"]["entities"]) == 2
    assert len(body["result"]["relationships"]) == 1
    assert body["result"]["warnings"] == []
    assert len(fake_llm_provider.calls) == 1


def test_extraction_handles_markdown_fenced_json(client, fake_llm_provider):
    fake_llm_provider.response = f"```json\n{VALID_LLM_RESPONSE}\n```"

    response = client.post("/api/v1/extraction", json={"text": "some incident text"})

    assert response.status_code == 200
    assert len(response.json()["result"]["entities"]) == 2


def test_extraction_empty_input_rejected(client, fake_llm_provider):
    response = client.post("/api/v1/extraction", json={"text": ""})

    assert response.status_code == 422
    assert response.json()["error"] == "ValidationError"
    # An empty string must fail validation before ever reaching the LLM provider.
    assert fake_llm_provider.calls == []


def test_extraction_invalid_request_missing_field(client, fake_llm_provider):
    response = client.post("/api/v1/extraction", json={})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ValidationError"
    assert "path" in body


def test_extraction_malformed_llm_response_does_not_crash_app(client, fake_llm_provider):
    fake_llm_provider.response = "this is not json at all {{{"

    response = client.post("/api/v1/extraction", json={"text": "some incident text"})

    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "ExtractionError"
    assert "path" in body


def test_extraction_empty_llm_response_does_not_crash_app(client, fake_llm_provider):
    fake_llm_provider.response = ""

    response = client.post("/api/v1/extraction", json={"text": "some incident text"})

    assert response.status_code == 502
    assert response.json()["error"] == "ExtractionError"


def test_extraction_non_object_json_response(client, fake_llm_provider):
    fake_llm_provider.response = json.dumps(["not", "an", "object"])

    response = client.post("/api/v1/extraction", json={"text": "some incident text"})

    assert response.status_code == 502
    assert response.json()["error"] == "ExtractionError"


def test_extraction_provider_failure_returns_503(client, fake_llm_provider):
    fake_llm_provider.raise_error = LLMProviderError("Groq is down")

    response = client.post("/api/v1/extraction", json={"text": "some incident text"})

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "LLMProviderError"
    assert "path" in body


def test_extraction_filters_invalid_entity_and_relationship_types(client, fake_llm_provider):
    fake_llm_provider.response = json.dumps(
        {
            "entities": [
                {"id": "e1", "name": "checkout-service", "type": "Service"},
                {"id": "e2", "name": "mystery-thing", "type": "NotARealType"},
            ],
            "relationships": [
                # references e2, which failed validation and was dropped
                {"source_id": "e1", "target_id": "e2", "type": "DEPENDS_ON"},
                # references an id that was never extracted at all
                {"source_id": "e1", "target_id": "ghost", "type": "CALLS"},
                # invalid relationship type
                {"source_id": "e1", "target_id": "e1", "type": "NOT_A_REAL_RELATIONSHIP"},
            ],
        }
    )

    response = client.post("/api/v1/extraction", json={"text": "some incident text"})

    assert response.status_code == 200
    body = response.json()
    assert [e["id"] for e in body["result"]["entities"]] == ["e1"]
    assert body["result"]["relationships"] == []
    assert len(body["result"]["warnings"]) >= 3


def test_extraction_missing_entities_and_relationships_keys(client, fake_llm_provider):
    """LLM returns valid but minimal JSON with no entities/relationships keys at all."""
    fake_llm_provider.response = json.dumps({})

    response = client.post("/api/v1/extraction", json={"text": "some incident text"})

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["entities"] == []
    assert body["result"]["relationships"] == []


def test_extraction_appears_in_openapi_schema(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/extraction" in paths
    assert "post" in paths["/api/v1/extraction"]


def test_health_reports_groq_not_configured_when_key_missing(client):
    """
    Explicitly overrides get_settings (rather than relying on the local .env
    having no key) so this test is correct regardless of whether GROQ_API_KEY
    happens to be set in the developer's own .env.
    """
    app.dependency_overrides[get_settings] = lambda: Settings(GROQ_API_KEY=None)
    try:
        response = client.get("/api/v1/health")
    finally:
        del app.dependency_overrides[get_settings]

    assert response.status_code == 200
    components = {c["name"]: c for c in response.json()["components"]}
    assert components["llm_provider_groq"]["status"] == "not_configured"


def test_health_reports_groq_ok_when_key_present(client):
    app.dependency_overrides[get_settings] = lambda: Settings(GROQ_API_KEY="test-key-123")
    try:
        response = client.get("/api/v1/health")
    finally:
        del app.dependency_overrides[get_settings]

    assert response.status_code == 200
    components = {c["name"]: c for c in response.json()["components"]}
    assert components["llm_provider_groq"]["status"] == "ok"