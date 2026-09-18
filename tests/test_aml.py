from fastapi.testclient import TestClient

from api.api import app


client = TestClient(app)


def test_aml_health():
    response = client.get("/api/v1/aml/health")
    assert response.status_code == 200
    body = response.json()
    assert body["domain"] == "aml"
    assert body["status"] == "not_implemented"


def test_aml_screen_not_implemented():
    response = client.post(
        "/api/v1/aml/screen",
        json={"party": {"full_name": "Jane Doe"}, "lists": ["sanctions"]},
    )
    assert response.status_code == 501
    assert response.json()["status"] == "error"
