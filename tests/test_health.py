import logging


def test_liveness(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_missing_models(client):
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["error"]
    assert body["models"]["face_detector"]["loaded"] is False
    assert body["models"]["ocr_extractor"]["loaded"] is False


def test_model_health_degraded_without_ml(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "face_detector" in body["models"]
    assert "liveness_detector" in body["models"]


def test_endpoint_logs_resource_and_result(client, caplog):
    caplog.set_level(logging.INFO, logger="api.api")
    response = client.get("/api/v1/liveness/challenge")
    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "resource=/challenge" in message
        and "path=/api/v1/liveness/challenge" in message
        and "result=" in message
        for message in messages
    )
