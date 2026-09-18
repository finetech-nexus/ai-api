# api/api.py

"""
FastAPI application for KYC verification service.
FIXED: Missing config import + response format for frontend
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, File, Request, UploadFile, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response as RawResponse

try:
    import cv2
    import numpy as np
except ImportError:  # tests / OpenAPI export without ML extras
    cv2 = None  # type: ignore
    np = None  # type: ignore

# ✅ FIX #1: Added missing import
from configs.config import config
from api.schemas import (
    KYCVerificationResponse,
    OCROnlyResponse,
    ErrorResponse,
    HealthCheckResponse,
    ModelStatus,
    VerificationStatus,
    OCRData,
    OCRFields,
    FaceMatchData,
    SimilarityMetrics,
    ChallengeResponse,
    ChallengeStatus,
    LivenessVerificationRequest,
    LivenessVerificationResponse,
    LivenessBatchRequest,
    LivenessBatchResponse,
    OCRRequest,
)
# ✅ LAZY IMPORT: Don't import ML libraries at module level
# They will be imported inside functions only when needed
# This allows the server to start even if ML libraries fail
from utils.logger import get_logger
from aml.router import router as aml_router

logger = get_logger(__name__, log_file="api.log")

# Global service instances (type hints will be resolved at runtime)
face_detector = None
face_matcher = None
ocr_extractor = None
liveness_detector = None  # Liveness detection service
ml_import_error: Optional[str] = None  # Track if ML libraries failed to import
model_errors: Dict[str, str] = {}

MODEL_LOADERS = (
    ("face_detector", "yunet"),
    ("face_matcher", "insightface"),
    ("ocr_extractor", "paddleocr"),
    ("liveness_detector", "mediapipe+haar"),
)

# Semaphore to limit concurrent processing
MAX_CONCURRENT = config.get("processing", "max_concurrent_requests", default=10)
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# ============================================================================
# Lifespan Event Handler
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    global face_detector, face_matcher, ocr_extractor, liveness_detector
    global ml_import_error, model_errors

    logger.info("Starting KYC Verification Service...")

    if os.environ.get("AI_API_SKIP_ML") == "1":
        ml_import_error = "AI_API_SKIP_ML=1"
        model_errors = {name: "AI_API_SKIP_ML=1" for name, _ in MODEL_LOADERS}
        logger.warning("AI_API_SKIP_ML=1: models not loaded")
        yield
        return

    factories = {}
    try:
        logger.info("Importing ML libraries...")
        import onnxruntime as ort

        logger.info("onnxruntime %s", ort.__version__)
        from app.services.face_detector_id import get_face_detector
        from app.services.face_matcher import get_face_matcher
        from app.services.ocr_extractor import get_ocr_extractor
        from app.services.liveness_detector import get_liveness_detector

        factories = {
            "face_detector": get_face_detector,
            "face_matcher": get_face_matcher,
            "ocr_extractor": get_ocr_extractor,
            "liveness_detector": get_liveness_detector,
        }
    except Exception as exc:
        ml_import_error = "import: {}: {}".format(type(exc).__name__, exc)
        model_errors = {name: ml_import_error for name, _ in MODEL_LOADERS}
        logger.exception("ML import failed")
        yield
        logger.info("Shutting down...")
        return

    for name, factory in factories.items():
        logger.info("Loading %s...", name)
        try:
            instance = await asyncio.to_thread(factory)
        except Exception as exc:
            model_errors[name] = "{}: {}".format(type(exc).__name__, exc)
            logger.exception("Failed to load %s", name)
            continue
        if name == "face_detector":
            face_detector = instance
        elif name == "face_matcher":
            face_matcher = instance
        elif name == "ocr_extractor":
            ocr_extractor = instance
        else:
            liveness_detector = instance
        logger.info("Loaded %s", name)

    if model_errors:
        ml_import_error = "; ".join(
            "{}: {}".format(name, err) for name, err in model_errors.items()
        )
        logger.error("Models not ready: %s", ml_import_error)
    else:
        logger.info("All models loaded successfully")

    yield
    logger.info("Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Nexus Bank AI API",
    description=(
        "KYC verification (face match, document OCR, liveness) from ./kyc. "
        "AML screening is reserved and returns 501 until models are added."
    ),
    version=config.get("project", "version", default="1.0.0"),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Liveness, readiness, and model status."},
        {"name": "KYC", "description": "Identity verification: face match, document OCR."},
        {"name": "OCR", "description": "Document field extraction."},
        {"name": "Liveness", "description": "Challenge-response liveness."},
        {"name": "AML", "description": "Anti-money-laundering screening (coming later)."},
    ],
)

# CORS configuration
cors_origins = config.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(aml_router, prefix="/api/v1/aml", tags=["AML"])

# Probe and docs traffic is noisy; log KYC/AML resources and their payloads.
_SKIP_LOG_PATHS = {"/health", "/ready", "/docs", "/redoc", "/openapi.json"}
_MAX_RESULT_CHARS = 2000


def _resource_name(path: str) -> str:
    parts = [segment for segment in path.split("/") if segment]
    return "/" + (parts[-1] if parts else "")


@app.middleware("http")
async def log_endpoint_result(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in _SKIP_LOG_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
        return response

    body = b""
    async for chunk in response.body_iterator:
        body += chunk

    result = body.decode("utf-8", errors="replace") or "-"
    if len(result) > _MAX_RESULT_CHARS:
        result = result[:_MAX_RESULT_CHARS] + "...[truncated]"

    logger.info(
        "resource=%s path=%s method=%s status=%s result=%s",
        _resource_name(path),
        path,
        request.method,
        response.status_code,
        result,
    )

    headers = dict(response.headers)
    headers.pop("content-length", None)
    return RawResponse(
        content=body,
        status_code=response.status_code,
        headers=headers,
        media_type=response.media_type,
        background=getattr(response, "background", None),
    )


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=str(exc.detail),
            details=None
        ).model_dump(mode="json")
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__}
        ).model_dump(mode="json")
    )


# ============================================================================
# Utility Functions
# ============================================================================

async def read_upload_file(upload_file: UploadFile) -> np.ndarray:
    """Read and validate uploaded image file."""
    if cv2 is None or np is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenCV is not installed",
        )
    max_size = config.max_upload_size
    content = await upload_file.read()
    
    try:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format. Supported: JPG, PNG"
            )
        
        max_dim = config.get("upload", "image_max_dimension", default=4096)
        h, w = image.shape[:2]
        
        # Log image dimensions for debugging
        logger.info(f"Image uploaded: {w}x{h} pixels, size: {len(content)/1024:.1f}KB")
        
        needs_resize = h > max_dim or w > max_dim or len(content) > max_size
        if needs_resize:
            scale = 1.0
            if h > max_dim or w > max_dim:
                scale = min(scale, max_dim / float(max(h, w)))
            if len(content) > max_size:
                scale = min(scale, 0.5)
            image = cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            logger.info(f"Image resized to {image.shape[1]}x{image.shape[0]}")
        
        # Validate minimum image resolution
        min_width, min_height = 320, 240
        h, w = image.shape[:2]
        if w < min_width or h < min_height:
            logger.warning(f"Image resolution too low: {w}x{h} (minimum: {min_width}x{min_height})")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image resolution too low ({w}x{h}). Please upload a higher resolution image (minimum {min_width}x{min_height})."
            )
        
        return image
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Image decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image"
        )


def determine_verification_status(face_verified: bool, ocr_confidence: float, face_confidence: float = 0.0) -> VerificationStatus:
    """
    Determine overall verification status.
    
    More lenient: If face match is decent (>= 0.35), approve even with low OCR confidence.
    This handles cases where OCR fails but face matching is reliable.
    """
    if not face_verified:
        return VerificationStatus.REJECTED
    
    # If face match confidence is decent (>= 0.35), approve even if OCR fails
    # This is more lenient for cases where OCR can't read the document but face match is reliable
    if face_confidence >= 0.35:
        return VerificationStatus.APPROVED
    
    # If OCR confidence is decent (>= 0.5), approve regardless of face match score
    if ocr_confidence >= 0.5:
        return VerificationStatus.APPROVED
    
    # If both are low, pending for manual review
    return VerificationStatus.PENDING


def calculate_confidence_score(
    face_confidence: float,
    ocr_confidence: float,
    face_verified: bool
) -> float:
    """
    Calculate overall confidence score (weighted: 60% face + 40% OCR).
    Used by frontend for display.
    """
    if not face_verified:
        return 0.0
    
    # Weighted average
    return (0.6 * face_confidence) + (0.4 * ocr_confidence)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", tags=["Health"], summary="Liveness probe")
async def liveness():
    """Process is up. Used by Kubernetes liveness probes."""
    return {"status": "ok"}


@app.get("/ready", tags=["Health"], summary="Readiness probe")
async def readiness():
    """Models are loaded. Used by Kubernetes readiness probes."""
    loaded = {
        "face_detector": face_detector,
        "face_matcher": face_matcher,
        "ocr_extractor": ocr_extractor,
        "liveness_detector": liveness_detector,
    }
    models = {
        name: {
            "loaded": instance is not None,
            "name": label,
            "error": None if instance is not None else model_errors.get(name, ml_import_error),
        }
        for (name, label), instance in zip(MODEL_LOADERS, loaded.values())
    }
    if all(item["loaded"] for item in models.values()):
        return {"status": "ready", "models": models}
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "error": ml_import_error,
            "models": models,
        },
    )


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint with model status.
    """
    models_status = {}
    
    # Check face detector
    models_status["face_detector"] = ModelStatus(
        loaded=face_detector is not None,
        name="yunet",
        error=None if face_detector is not None else model_errors.get("face_detector", ml_import_error),
    )
    
    # Check face matcher
    models_status["face_matcher"] = ModelStatus(
        loaded=face_matcher is not None,
        name="insightface",
        error=None if face_matcher is not None else model_errors.get("face_matcher", ml_import_error),
    )
    
    # Check OCR extractor
    models_status["ocr_extractor"] = ModelStatus(
        loaded=ocr_extractor is not None,
        name="paddleocr",
        error=None if ocr_extractor is not None else model_errors.get("ocr_extractor", ml_import_error),
    )
    
    # Check liveness detector
    models_status["liveness_detector"] = ModelStatus(
        loaded=liveness_detector is not None,
        name="mediapipe+haar",
        error=None if liveness_detector is not None else model_errors.get("liveness_detector", ml_import_error),
    )
    
    all_loaded = all(m.loaded for m in models_status.values())
    
    return HealthCheckResponse(
        status="healthy" if all_loaded else "degraded",
        version=config.get("project", "version", default="1.0.0"),
        models=models_status
    )


@app.post(
    "/api/v1/kyc/verify",
    response_model=KYCVerificationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["KYC"]
)
async def verify_kyc(
    id_document: UploadFile = File(..., description="ID card/passport image"),
    selfie_image: UploadFile = File(..., description="Selfie photo")
):
    """
    Complete KYC verification: face detection + matching + OCR.
    """
    start_time = time.time()
    
    if not all([face_detector, face_matcher, ocr_extractor]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready. Models still loading."
        )
    
    async with processing_semaphore:
        try:
            logger.info("Reading uploaded files...")
            id_image = await read_upload_file(id_document)
            selfie_img = await read_upload_file(selfie_image)
            
            # Detect faces
            logger.info("Detecting faces...")
            id_face_result, selfie_face_result = await asyncio.gather(
                asyncio.to_thread(face_detector.detect_and_extract, id_image),
                asyncio.to_thread(face_detector.detect_and_extract, selfie_img)
            )
            
            if id_face_result is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected in ID document. Please ensure: (1) Face is clearly visible, (2) Image is high resolution (minimum 640x480), (3) Face is not too small or far from camera, (4) Good lighting without glare."
                )
            
            if selfie_face_result is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected in selfie image. Please ensure: (1) Face is clearly visible and centered, (2) Image is high resolution, (3) Good lighting, (4) Face is not too small."
                )
            
            # ✅ OPTIMIZATION: Run face matching and OCR in parallel
            logger.info("Running face matching and OCR in parallel...")
            try:
                match_result, ocr_result = await asyncio.gather(
                    asyncio.to_thread(
                        face_matcher.verify,
                        id_face_result.face_crop,
                        selfie_face_result.face_crop
                    ),
                    asyncio.to_thread(
                        ocr_extractor.extract_structured,
                        id_image
                    ),
                    return_exceptions=True  # Don't fail entire request if one fails
                )
                
                # Check if face matching failed
                if isinstance(match_result, Exception):
                    logger.error(f"Face matching failed: {match_result}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Face matching failed: {str(match_result)}"
                    )
                
                # Check if OCR failed (PaddlePaddle segfault protection)
                if isinstance(ocr_result, Exception):
                    logger.error(f"OCR extraction failed: {ocr_result}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="OCR extraction failed. Please try with a clearer document image or try again."
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Processing failed: {str(e)}"
                )
            
            # Determine verification status
            verification_status = determine_verification_status(
                face_verified=match_result.verified,
                ocr_confidence=ocr_result.confidence,
                face_confidence=match_result.confidence
            )
            
            # ✅ FIX #2: Calculate overall confidence score for frontend
            confidence_score = calculate_confidence_score(
                face_confidence=match_result.confidence,
                ocr_confidence=ocr_result.confidence,
                face_verified=match_result.verified
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Log detailed face matching info for debugging
            logger.info(f"Face Matching Details:")
            logger.info(f"  - cosine_similarity: {match_result.cosine_similarity:.4f} ({match_result.cosine_similarity*100:.1f}%)")
            logger.info(f"  - confidence (normalized_score): {match_result.confidence:.4f} ({match_result.confidence*100:.1f}%)")
            logger.info(f"  - threshold_used: {match_result.threshold_used:.4f} ({match_result.threshold_used*100:.1f}%)")
            logger.info(f"  - verified: {match_result.verified} (cosine_sim >= threshold: {match_result.cosine_similarity:.4f} >= {match_result.threshold_used:.4f})")
            logger.info(f"✓ Verification complete: {verification_status.value} (confidence: {confidence_score:.2f})")
            
            # Convert OCRResult to OCRData (Pydantic model)
            # All fields are Optional, so missing fields will be None - this handles documents with incomplete data
            ocr_fields = OCRFields(
                full_name=ocr_result.full_name,
                date_of_birth=ocr_result.date_of_birth,
                document_number=ocr_result.document_number,
                nationality=ocr_result.nationality,
                issue_date=ocr_result.issue_date,
                expiry_date=ocr_result.expiry_date,
                place_of_birth=ocr_result.place_of_birth,
                address=ocr_result.address,
                gender=ocr_result.gender
            )
            
            ocr_data = OCRData(
                document_type=ocr_result.document_type,
                confidence=ocr_result.confidence,
                extracted_text=ocr_result.extracted_text,
                fields=ocr_fields
            )
            
            return KYCVerificationResponse(
                verification_status=verification_status,
                confidence_score=confidence_score,
                face_match_score=match_result.confidence,  # This is normalized_score, not cosine_similarity
                ocr_data=ocr_data,
                processing_time_ms=processing_time_ms,
                face_verification_details=match_result.to_dict()
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Verification error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Verification failed: {str(e)}"
            )


async def run_ocr(doc_image) -> OCROnlyResponse:
    if ocr_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service not ready. Models still loading."
        )
    start_time = time.time()
    async with processing_semaphore:
        try:
            logger.info("Extracting OCR...")
            ocr_result = await asyncio.to_thread(
                ocr_extractor.extract_structured,
                doc_image
            )
            processing_time_ms = int((time.time() - start_time) * 1000)
            confidence = max(0.0, min(1.0, float(ocr_result.confidence or 0.0)))
            response = OCROnlyResponse(
                ocr_data=OCRData(
                    document_type=ocr_result.document_type,
                    confidence=confidence,
                    extracted_text=ocr_result.extracted_text,
                    fields=OCRFields(**{
                        k: v for k, v in ocr_result.to_dict().items()
                        if k in OCRFields.model_fields
                    })
                ),
                processing_time_ms=processing_time_ms
            )
            logger.info(
                "OCR extraction complete (%sms, type=%s, confidence=%s)",
                processing_time_ms,
                ocr_result.document_type,
                confidence,
            )
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"OCR error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR extraction failed: {str(e)}"
            )


# ============================================================================
# Liveness Detection Endpoints
# ============================================================================

def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 image string to numpy array.
    
    Args:
        base64_str: Base64-encoded image (with or without data URI prefix)
    
    Returns:
        Decoded image as numpy array (BGR format)
    """
    import base64
    from io import BytesIO

    if cv2 is None or np is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenCV is not installed",
        )
    
    # Remove data URI prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]
    
    try:
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            return image

        from PIL import Image
        pil_image = Image.open(BytesIO(image_data))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode base64 image: {str(e)}"
        )


async def read_ocr_image(request: Request):
    """Accept JSON `{document: base64}` or multipart field `document` (string or file)."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = None
        raw = None
        if isinstance(body, dict):
            raw = body.get("document") or body.get("image")
        if not isinstance(raw, str) or not raw.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="document image is required",
            )
        logger.info("OCR JSON request received (%s chars)", len(raw))
        return decode_base64_image(raw)

    form = await request.form()
    item = form.get("document") or form.get("image")
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document image is required",
        )
    if isinstance(item, str) and item.strip():
        logger.info("OCR form-data string received (%s chars)", len(item))
        return decode_base64_image(item)
    if hasattr(item, "read"):
        logger.info("OCR multipart file received")
        return await read_upload_file(item)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="document image is required",
    )


@app.post(
    "/api/v1/ocr/extract",
    response_model=OCROnlyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["OCR"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": OCRRequest.model_json_schema()},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {"document": {"type": "string"}},
                        "required": ["document"],
                    }
                },
            },
        }
    },
)
@app.post(
    "/api/v1/kyc/ocr",
    response_model=OCROnlyResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["KYC"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": OCRRequest.model_json_schema()},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {"document": {"type": "string"}},
                        "required": ["document"],
                    }
                },
            },
        }
    },
)
async def extract_ocr(request: Request):
    """OCR from JSON base64 or multipart form field `document`."""
    return await run_ocr(await read_ocr_image(request))


@app.get(
    "/api/v1/liveness/challenge",
    response_model=ChallengeResponse,
    tags=["Liveness"]
)
@app.get(
    "/api/v1/kyc/liveness/challenge",
    response_model=ChallengeResponse,
    tags=["KYC"],
)
async def generate_challenge(types: Optional[str] = Query(None, description="Comma-separated challenge types, e.g. turn_left,turn_right")):
    """
    Generate a new liveness challenge.
    Returns a challenge that the user must complete (blink, turn left, turn right).
    Pass types=turn_left,turn_right to request a left-then-right head turn.
    """
    try:
        from app.services.liveness_challenges import ChallengeType, get_challenge_generator
        
        generator = get_challenge_generator()
        requested = []
        if types:
            for raw in types.split(","):
                key = raw.strip().lower()
                try:
                    requested.append(ChallengeType(key))
                except ValueError:
                    logger.warning(f"Ignoring unknown liveness type: {key}")
        if requested:
            challenge = generator.generate_challenge(challenge_types=requested, num_challenges=len(requested))
        else:
            challenge = generator.generate_challenge()
        
        logger.info(f"Generated challenge: {challenge.challenge_type.value} (ID: {challenge.challenge_id})")
        
        return ChallengeResponse(**challenge.to_dict())
    
    except Exception as e:
        logger.error(f"Challenge generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate challenge: {str(e)}"
        )


@app.post(
    "/api/v1/liveness/verify",
    response_model=LivenessVerificationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Liveness"]
)
@app.post(
    "/api/v1/kyc/liveness/verify",
    response_model=LivenessVerificationResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["KYC"],
)
async def verify_liveness_challenge(request: LivenessVerificationRequest):
    """
    Verify a liveness challenge with captured frames.
    Processes frames and validates against the challenge requirements.
    """
    start_time = time.time()
    
    if liveness_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Liveness detector not available. Service may still be loading."
        )
    
    async with processing_semaphore:
        try:
            # Validate request
            if not request.frames or len(request.frames) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No frames provided"
                )
            
            min_frames = config.get("liveness", "detection", "min_frames", default=10)
            if len(request.frames) < min_frames:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Not enough frames. Minimum: {min_frames}, received: {len(request.frames)}"
                )
            
            logger.info(f"Verifying challenge {request.challenge_id} with {len(request.frames)} frames...")
            
            # Decode base64 frames
            decoded_frames = []
            for i, frame_str in enumerate(request.frames):
                try:
                    frame = decode_base64_image(frame_str)
                    decoded_frames.append(frame)
                except Exception as e:
                    logger.warning(f"Failed to decode frame {i}: {e}")
                    # Continue with other frames
            
            if len(decoded_frames) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to decode any frames"
                )
            
            # Verify challenge
            status_result, message, results = await asyncio.to_thread(
                liveness_detector.verify_challenge,
                request.challenge_id,
                decoded_frames,
                0,  # initial_counter
                0   # initial_total
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"Challenge verification: {status_result.value} - {message}")
            
            return LivenessVerificationResponse(
                challenge_id=request.challenge_id,
                status=status_result,
                message=message,
                detection_results=results.get("detection_results", {}),
                processing_time_ms=processing_time_ms
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Liveness verification error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Liveness verification failed: {str(e)}"
            )


@app.post(
    "/api/v1/liveness/detect",
    response_model=LivenessBatchResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Liveness"]
)
@app.post(
    "/api/v1/kyc/liveness/detect",
    response_model=LivenessBatchResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["KYC"],
)
async def detect_liveness_batch(request: LivenessBatchRequest):
    """
    Perform batch liveness detection without challenge.
    Useful for continuous detection or testing.
    """
    start_time = time.time()
    
    if liveness_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Liveness detector not available. Service may still be loading."
        )
    
    async with processing_semaphore:
        try:
            # Validate request
            if not request.frames or len(request.frames) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No frames provided"
                )
            
            logger.info(f"Processing batch liveness detection with {len(request.frames)} frames...")
            
            # Decode base64 frames
            decoded_frames = []
            for i, frame_str in enumerate(request.frames):
                try:
                    frame = decode_base64_image(frame_str)
                    decoded_frames.append(frame)
                except Exception as e:
                    logger.warning(f"Failed to decode frame {i}: {e}")
                    # Continue with other frames
            
            if len(decoded_frames) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to decode any frames"
                )
            
            # Detect liveness
            batch_results = await asyncio.to_thread(
                liveness_detector.detect_batch,
                decoded_frames,
                0,  # initial_counter
                request.initial_blink_count  # initial_total
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return LivenessBatchResponse(
                total_blinks=batch_results.get("total_blinks", 0),
                final_blink_count=batch_results.get("final_blink_count", 0),
                orientations=batch_results.get("orientations", []),
                face_detection_ratio=batch_results.get("face_detection_ratio", 0.0),
                results=batch_results.get("results", []),
                frame_count=batch_results.get("frame_count", len(decoded_frames)),
                processing_time_ms=processing_time_ms
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch liveness detection error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch liveness detection failed: {str(e)}"
            )
