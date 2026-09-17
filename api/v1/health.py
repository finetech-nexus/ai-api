"""Health and readiness endpoints."""

from fastapi import APIRouter, Response, status

from core.settings import get_settings
from domains.kyc.runtime import runtime
from domains.kyc.schemas import HealthCheckResponse, ModelStatus

router = APIRouter(tags=["Health"])

MODEL_LABELS = {
    "face_detector": "yunet",
    "face_matcher": "insightface",
    "ocr_extractor": "paddleocr",
    "liveness_detector": "mediapipe+haar",
}


@router.get("/health", summary="Liveness probe")
async def liveness():
    """Process is up. Used by Kubernetes liveness probes."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def readiness(response: Response):
    """Models are loaded. Used by Kubernetes readiness probes."""
    if runtime.models_ready():
        return {"status": "ready"}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "not_ready",
        "error": runtime.ml_import_error,
    }


@router.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    summary="KYC model health",
)
async def model_health():
    settings = get_settings()
    models = {}
    for attr, name in MODEL_LABELS.items():
        loaded = getattr(runtime, attr) is not None
        models[attr] = ModelStatus(
            loaded=loaded,
            name=name,
            error=None if loaded else runtime.model_errors.get(attr, runtime.ml_import_error),
        )
    all_loaded = all(model.loaded for model in models.values())
    return HealthCheckResponse(
        status="healthy" if all_loaded else "degraded",
        version=settings.app_version,
        models=models,
    )
