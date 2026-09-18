EXPECTED_PATHS = {
    "/health",
    "/ready",
    "/api/v1/health",
    "/api/v1/kyc/verify",
    "/api/v1/kyc/ocr",
    "/api/v1/kyc/liveness/challenge",
    "/api/v1/kyc/liveness/verify",
    "/api/v1/kyc/liveness/detect",
    "/api/v1/ocr/extract",
    "/api/v1/liveness/challenge",
    "/api/v1/liveness/verify",
    "/api/v1/liveness/detect",
    "/api/v1/aml/health",
    "/api/v1/aml/screen",
}


def test_openapi_exposes_kyc_and_aml(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert spec["info"]["title"] == "Nexus Bank AI API"
    paths = set(spec["paths"].keys())
    missing = EXPECTED_PATHS - paths
    assert not missing, f"missing OpenAPI paths: {missing}"
    assert spec["paths"]["/api/v1/kyc/verify"]["post"]["tags"] == ["KYC"]
    assert spec["paths"]["/api/v1/aml/screen"]["post"]["tags"] == ["AML"]
    ocr_content = spec["paths"]["/api/v1/kyc/ocr"]["post"]["requestBody"]["content"]
    assert "application/json" in ocr_content


def test_kyc_ocr_accepts_json_document(client):
    response = client.post(
        "/api/v1/kyc/ocr",
        json={"document": "data:image/jpeg;base64,QQ=="},
    )
    assert response.status_code != 422
    detail = str(response.json())
    assert "UploadFile" not in detail
    assert "Expected UploadFile" not in detail
