"""KYC domain operations wrapping the vendored ML engine."""

from __future__ import annotations

import asyncio
import time

from fastapi import HTTPException, status

from domains.kyc.runtime import runtime
from domains.kyc.schemas import (
    OCRData,
    OCRFields,
    OCROnlyResponse,
    KYCVerificationResponse,
    VerificationStatus,
)


def determine_verification_status(
    face_verified: bool,
    ocr_confidence: float,
    face_confidence: float = 0.0,
) -> VerificationStatus:
    if not face_verified:
        return VerificationStatus.REJECTED
    if face_confidence >= 0.35:
        return VerificationStatus.APPROVED
    if ocr_confidence >= 0.5:
        return VerificationStatus.APPROVED
    return VerificationStatus.PENDING


def calculate_confidence_score(
    face_confidence: float,
    ocr_confidence: float,
    face_verified: bool,
) -> float:
    if not face_verified:
        return 0.0
    return (0.6 * face_confidence) + (0.4 * ocr_confidence)


def require_kyc_models() -> None:
    if not all([runtime.face_detector, runtime.face_matcher, runtime.ocr_extractor]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="KYC models are not ready.",
        )


def require_liveness() -> None:
    if runtime.liveness_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Liveness detector is not ready.",
        )


async def verify_identity(id_image, selfie_img) -> KYCVerificationResponse:
    require_kyc_models()
    start_time = time.time()

    async with runtime.processing_semaphore:
        id_face_result, selfie_face_result = await asyncio.gather(
            asyncio.to_thread(runtime.face_detector.detect_and_extract, id_image),
            asyncio.to_thread(runtime.face_detector.detect_and_extract, selfie_img),
        )

        if id_face_result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No face detected in ID document. Please ensure: "
                    "(1) Face is clearly visible, (2) Image is high resolution (minimum 640x480), "
                    "(3) Face is not too small, (4) Good lighting without glare."
                ),
            )

        if selfie_face_result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No face detected in selfie image. Please ensure: "
                    "(1) Face is clearly visible and centered, (2) Image is high resolution, "
                    "(3) Good lighting, (4) Face is not too small."
                ),
            )

        match_result, ocr_result = await asyncio.gather(
            asyncio.to_thread(
                runtime.face_matcher.verify,
                id_face_result.face_crop,
                selfie_face_result.face_crop,
            ),
            asyncio.to_thread(runtime.ocr_extractor.extract_structured, id_image),
            return_exceptions=True,
        )

        if isinstance(match_result, Exception):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Face matching failed: {match_result}",
            )
        if isinstance(ocr_result, Exception):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OCR extraction failed. Please try with a clearer document image.",
            )

        verification_status = determine_verification_status(
            face_verified=match_result.verified,
            ocr_confidence=ocr_result.confidence,
            face_confidence=match_result.confidence,
        )
        confidence_score = calculate_confidence_score(
            face_confidence=match_result.confidence,
            ocr_confidence=ocr_result.confidence,
            face_verified=match_result.verified,
        )

        ocr_data = OCRData(
            document_type=ocr_result.document_type,
            confidence=ocr_result.confidence,
            extracted_text=ocr_result.extracted_text,
            fields=OCRFields(
                full_name=ocr_result.full_name,
                date_of_birth=ocr_result.date_of_birth,
                document_number=ocr_result.document_number,
                nationality=ocr_result.nationality,
                issue_date=ocr_result.issue_date,
                expiry_date=ocr_result.expiry_date,
                place_of_birth=ocr_result.place_of_birth,
                address=ocr_result.address,
                gender=ocr_result.gender,
            ),
        )

        return KYCVerificationResponse(
            verification_status=verification_status,
            confidence_score=confidence_score,
            face_match_score=match_result.confidence,
            ocr_data=ocr_data,
            processing_time_ms=int((time.time() - start_time) * 1000),
            face_verification_details=match_result.to_dict(),
        )


async def extract_ocr(doc_image) -> OCROnlyResponse:
    if runtime.ocr_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service is not ready.",
        )
    start_time = time.time()
    async with runtime.processing_semaphore:
        ocr_result = await asyncio.to_thread(
            runtime.ocr_extractor.extract_structured,
            doc_image,
        )
        return OCROnlyResponse(
            ocr_data=OCRData(
                document_type=ocr_result.document_type,
                confidence=ocr_result.confidence,
                extracted_text=ocr_result.extracted_text,
                fields=OCRFields(
                    **{
                        k: v
                        for k, v in ocr_result.to_dict().items()
                        if k in OCRFields.model_fields
                    }
                ),
            ),
            processing_time_ms=int((time.time() - start_time) * 1000),
        )
