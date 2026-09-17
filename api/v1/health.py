"""Health and readiness endpoints."""

from fastapi import APIRouter, Response, status

from core.settings import get_settings
from domains.kyc.runtime import runtime
from domains.kyc.schemas import HealthCheckResponse, ModelStatus

router = APIRouter(tags=["Health"])


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
    models = {
        "face_detector": ModelStatus(
            loaded=runtime.face_detector is not None,
            name="yunet",
            error=runtime.ml_import_error if runtime.face_detector is None else None,
        ),
        "face_matcher": ModelStatus(
            loaded=runtime.face_matcher is not None,
            name="insightface",
            error=runtime.ml_import_error if runtime.face_matcher is None else None,
        ),
        "ocr_extractor": ModelStatus(
            loaded=runtime.ocr_extractor is not None,
            name="paddleocr",
            error=runtime.ml_import_error if runtime.ocr_extractor is None else None,
        ),
        "liveness_detector": ModelStatus(
            loaded=runtime.liveness_detector is not None,
            name="mediapipe+haar",
            error=runtime.ml_import_error if runtime.liveness_detector is None else None,
        ),
    }
    all_loaded = all(model.loaded for model in models.values())
    return HealthCheckResponse(
        status="healthy" if all_loaded else "degraded",
        version=settings.app_version,
        models=models,
    )
