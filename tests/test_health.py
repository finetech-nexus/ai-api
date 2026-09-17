from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_liveness():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_health_degraded_without_ml():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "face_detector" in body["models"]
    assert "liveness_detector" in body["models"]
