"""Lazy-loaded KYC ML runtime (YuNet, InsightFace, PaddleOCR, liveness)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.settings import get_settings

logger = logging.getLogger(__name__)

MODEL_NAMES = ("face_detector", "face_matcher", "ocr_extractor", "liveness_detector")


@dataclass
class KycRuntime:
    face_detector: Any = None
    face_matcher: Any = None
    ocr_extractor: Any = None
    liveness_detector: Any = None
    ml_import_error: Optional[str] = None
    model_errors: Dict[str, str] = field(default_factory=dict)
    processing_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(get_settings().max_concurrent_requests)
    )

    def models_ready(self) -> bool:
        return all(getattr(self, name) is not None for name in MODEL_NAMES)

    async def load(self) -> None:
        if get_settings().AI_API_SKIP_ML:
            self.ml_import_error = "AI_API_SKIP_ML=1"
            self.model_errors = {name: "AI_API_SKIP_ML=1" for name in MODEL_NAMES}
            logger.warning("AI_API_SKIP_ML=1: KYC models not loaded, /ready stays not_ready")
            return

        stage = "import"
        try:
            import onnxruntime as ort  # noqa: F401

            from app.services.face_detector_id import get_face_detector
            from app.services.face_matcher import get_face_matcher
            from app.services.ocr_extractor import get_ocr_extractor
            from app.services.liveness_detector import get_liveness_detector

            loaders = (
                ("face_detector", get_face_detector),
                ("face_matcher", get_face_matcher),
                ("ocr_extractor", get_ocr_extractor),
                ("liveness_detector", get_liveness_detector),
            )
            for stage, factory in loaders:
                logger.info("Loading %s", stage)
                started = time.monotonic()
                setattr(self, stage, await asyncio.to_thread(factory))
                logger.info("Loaded %s in %.1fs", stage, time.monotonic() - started)

            self.ml_import_error = None
            self.model_errors = {}
            logger.info("KYC models ready")
        except Exception as exc:
            detail = "{}: {}".format(type(exc).__name__, exc)
            self.ml_import_error = "{}: {}".format(stage, detail)
            logger.exception("KYC model load failed at stage %r", stage)
            # Loading is sequential and all-or-nothing, so only `stage` actually
            # failed; the rest were never attempted.
            self.model_errors = {
                name: detail if name == stage else "not loaded ({} failed)".format(stage)
                for name in MODEL_NAMES
            }
            for name in MODEL_NAMES:
                setattr(self, name, None)


runtime = KycRuntime()
