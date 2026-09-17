"""Lazy-loaded KYC ML runtime (YuNet, InsightFace, PaddleOCR, liveness)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from core.settings import get_settings


@dataclass
class KycRuntime:
    face_detector: Any = None
    face_matcher: Any = None
    ocr_extractor: Any = None
    liveness_detector: Any = None
    ml_import_error: Optional[str] = None
    processing_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(get_settings().max_concurrent_requests)
    )

    def models_ready(self) -> bool:
        return all(
            [
                self.face_detector is not None,
                self.face_matcher is not None,
                self.ocr_extractor is not None,
                self.liveness_detector is not None,
            ]
        )

    async def load(self) -> None:
        if get_settings().AI_API_SKIP_ML:
            self.ml_import_error = "AI_API_SKIP_ML=1"
            return
        try:
            import onnxruntime as ort  # noqa: F401

            from app.services.face_detector_id import get_face_detector
            from app.services.face_matcher import get_face_matcher
            from app.services.ocr_extractor import get_ocr_extractor
            from app.services.liveness_detector import get_liveness_detector

            self.face_detector = await asyncio.to_thread(get_face_detector)
            self.face_matcher = await asyncio.to_thread(get_face_matcher)
            self.ocr_extractor = await asyncio.to_thread(get_ocr_extractor)
            self.liveness_detector = await asyncio.to_thread(get_liveness_detector)
            self.ml_import_error = None
        except Exception as exc:
            self.ml_import_error = str(exc)
            self.face_detector = None
            self.face_matcher = None
            self.ocr_extractor = None
            self.liveness_detector = None


runtime = KycRuntime()
