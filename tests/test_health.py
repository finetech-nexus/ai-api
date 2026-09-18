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
