"""API-layer tests for app/api/v1/endpoints/datasets.py (Phase 3, Part 7)."""


def test_list_cases_endpoint(dataset_client):
    response = dataset_client.get("/api/v1/datasets")

    assert response.status_code == 200
    cases = response.json()
    assert len(cases) == 2
    assert {c["case_id"] for c in cases} == {"case-0001", "case-0002"}


def test_list_cases_endpoint_never_leaks_directory_names(dataset_client):
    response = dataset_client.get("/api/v1/datasets")
    text = response.text

    assert "cartservice" not in text
    assert "orders" not in text
    assert "_cpu_" not in text
    assert "_delay_" not in text


def test_get_case_endpoint(dataset_client):
    cases = dataset_client.get("/api/v1/datasets").json()
    case_id = cases[0]["case_id"]

    response = dataset_client.get(f"/api/v1/datasets/{case_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["case_id"] == case_id
    assert "root_cause_service" not in body
    assert "fault_type" not in body


def test_get_unknown_case_returns_404(dataset_client):
    response = dataset_client.get("/api/v1/datasets/case-9999")
    assert response.status_code == 404
    assert response.json()["error"] == "DatasetCaseNotFoundError"


def test_telemetry_summary_endpoint(dataset_client):
    cases = dataset_client.get("/api/v1/datasets").json()
    case_id = next(c["case_id"] for c in cases if c["system"] == "online-boutique")

    response = dataset_client.get(f"/api/v1/datasets/{case_id}/telemetry/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["row_counts"]["metrics"] == 2  # cutoff enforced


def test_telemetry_rows_endpoint_respects_max_rows(dataset_client):
    cases = dataset_client.get("/api/v1/datasets").json()
    case_id = next(c["case_id"] for c in cases if c["system"] == "sock-shop")

    response = dataset_client.get(
        f"/api/v1/datasets/{case_id}/telemetry", params={"max_rows": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["metrics"]) == 1
    assert len(body["logs"]) == 1
    assert len(body["traces"]) == 1


def test_telemetry_rows_endpoint_excludes_future_data(dataset_client):
    cases = dataset_client.get("/api/v1/datasets").json()
    case_id = next(c["case_id"] for c in cases if c["system"] == "sock-shop")

    response = dataset_client.get(f"/api/v1/datasets/{case_id}/telemetry")
    body = response.json()

    messages = [row["message"] for row in body["logs"]]
    assert "elevated response time observed - excluded from any diagnostic window" not in messages


def test_datasets_appear_in_openapi_schema(dataset_client):
    response = dataset_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/datasets" in paths
    assert "/api/v1/datasets/{case_id}" in paths