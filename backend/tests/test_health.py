from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["service"] == "RootCause-CC"


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["components"]) == 3
    assert {c["name"] for c in body["components"]} == {
        "neo4j_graph_db",
        "vector_store",
        "llm_provider_groq",
    }


def test_404_returns_consistent_error_shape() -> None:
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "HTTPException"
    assert "path" in body