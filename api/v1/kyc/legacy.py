"""Legacy path aliases matching the previous kyc/api ML backend."""

from fastapi import APIRouter, File, UploadFile

from api.v1.kyc.router import (
    detect_liveness,
    extract_ocr,
    generate_challenge,
    verify_liveness,
)
from core.errors import ErrorResponse
from domains.kyc.schemas import OCROnlyResponse

legacy_router = APIRouter()

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@legacy_router.post(
    "/ocr/extract",
    response_model=OCROnlyResponse,
    responses=ERROR_RESPONSES,
    summary="Extract document fields (legacy path)",
    include_in_schema=True,
)
async def extract_ocr_legacy(
    document: UploadFile = File(..., description="Document image for OCR"),
):
    return await extract_ocr(document=document)


legacy_router.add_api_route(
    "/liveness/challenge",
    generate_challenge,
    methods=["GET"],
    summary="Create a liveness challenge (legacy path)",
)
legacy_router.add_api_route(
    "/liveness/verify",
    verify_liveness,
    methods=["POST"],
    summary="Verify a liveness challenge (legacy path)",
)
legacy_router.add_api_route(
    "/liveness/detect",
    detect_liveness,
    methods=["POST"],
    summary="Batch liveness detection (legacy path)",
)
