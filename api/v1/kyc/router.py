"""KYC REST resources — same contract as the previous kyc/api ML backend."""

from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from core.errors import ErrorResponse
from core.images import decode_base64_image, read_upload_file
from domains.kyc import service
from domains.kyc.runtime import runtime
from domains.kyc.schemas import (
    ChallengeResponse,
    KYCVerificationResponse,
    LivenessBatchRequest,
    LivenessBatchResponse,
    LivenessVerificationRequest,
    LivenessVerificationResponse,
    OCROnlyResponse,
)

router = APIRouter()

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post(
    "/verify",
    response_model=KYCVerificationResponse,
    responses=ERROR_RESPONSES,
    summary="Verify identity",
    description="Face detection + face match + OCR on an ID document and a selfie.",
)
async def verify_kyc(
    id_document: UploadFile = File(..., description="ID card or passport image"),
    selfie_image: UploadFile = File(..., description="Selfie photo"),
):
    id_image = await read_upload_file(id_document)
    selfie_img = await read_upload_file(selfie_image)
    return await service.verify_identity(id_image, selfie_img)


@router.post(
    "/ocr",
    response_model=OCROnlyResponse,
    responses=ERROR_RESPONSES,
    summary="Extract document fields",
)
async def extract_ocr(
    document: UploadFile = File(..., description="Document image for OCR"),
):
    doc_image = await read_upload_file(document)
    return await service.extract_ocr(doc_image)


@router.get(
    "/liveness/challenge",
    response_model=ChallengeResponse,
    summary="Create a liveness challenge",
)
async def generate_challenge(
    types: Optional[str] = Query(
        None,
        description="Comma-separated challenge types, e.g. turn_left,turn_right",
    ),
):
    from app.services.liveness_challenges import ChallengeType, get_challenge_generator

    generator = get_challenge_generator()
    requested = []
    if types:
        for raw in types.split(","):
            key = raw.strip().lower()
            try:
                requested.append(ChallengeType(key))
            except ValueError:
                continue
    if requested:
        challenge = generator.generate_challenge(
            challenge_types=requested,
            num_challenges=len(requested),
        )
    else:
        challenge = generator.generate_challenge()
    return ChallengeResponse(**challenge.to_dict())


@router.post(
    "/liveness/verify",
    response_model=LivenessVerificationResponse,
    responses=ERROR_RESPONSES,
    summary="Verify a liveness challenge",
)
async def verify_liveness(request: LivenessVerificationRequest):
    import asyncio
    import time

    from configs.config import config

    service.require_liveness()
    if not request.frames:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No frames provided")

    min_frames = config.get("liveness", "detection", "min_frames", default=10)
    if len(request.frames) < min_frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not enough frames. Minimum: {min_frames}, received: {len(request.frames)}",
        )

    start_time = time.time()
    async with runtime.processing_semaphore:
        decoded_frames = []
        for frame_str in request.frames:
            try:
                decoded_frames.append(decode_base64_image(frame_str))
            except HTTPException:
                continue
        if not decoded_frames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode any frames",
            )

        status_result, message, results = await asyncio.to_thread(
            runtime.liveness_detector.verify_challenge,
            request.challenge_id,
            decoded_frames,
            0,
            0,
        )
        return LivenessVerificationResponse(
            challenge_id=request.challenge_id,
            status=status_result,
            message=message,
            detection_results=results.get("detection_results", {}),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )


@router.post(
    "/liveness/detect",
    response_model=LivenessBatchResponse,
    responses=ERROR_RESPONSES,
    summary="Batch liveness detection",
)
async def detect_liveness(request: LivenessBatchRequest):
    import asyncio
    import time

    service.require_liveness()
    if not request.frames:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No frames provided")

    start_time = time.time()
    async with runtime.processing_semaphore:
        decoded_frames = []
        for frame_str in request.frames:
            try:
                decoded_frames.append(decode_base64_image(frame_str))
            except HTTPException:
                continue
        if not decoded_frames:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode any frames",
            )

        batch_results = await asyncio.to_thread(
            runtime.liveness_detector.detect_batch,
            decoded_frames,
            0,
            request.initial_blink_count,
        )
        return LivenessBatchResponse(
            total_blinks=batch_results.get("total_blinks", 0),
            final_blink_count=batch_results.get("final_blink_count", 0),
            orientations=batch_results.get("orientations", []),
            face_detection_ratio=batch_results.get("face_detection_ratio", 0.0),
            results=batch_results.get("results", []),
            frame_count=batch_results.get("frame_count", len(decoded_frames)),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )
