# app/services/face_detector_id.py

"""
YuNet Face Detector - Optimized for KYC verification
FIXED: Separate log file, improved error messages
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import threading

from configs.config import config
from utils.logger import get_logger

# ✅ FIX: Use separate log file for face detection
logger = get_logger(__name__, log_file="face_detector.log")


class FaceDetectionResult:
    """Encapsulates face detection result"""
    
    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        landmarks: np.ndarray,
        face_crop: Optional[np.ndarray] = None
    ):
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.landmarks = landmarks  # 5 points: [right_eye, left_eye, nose, right_mouth, left_mouth]
        self.face_crop = face_crop
    
    def to_dict(self) -> dict:
        """Convert to serializable dict"""
        return {
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "landmarks": self.landmarks.tolist(),
            "has_crop": self.face_crop is not None
        }


class YuNetFaceDetector:
    """
    YuNet face detector using OpenCV's FaceDetectorYN.
    Optimized for:
    - ID document face detection (single face, may be small/rotated)
    - Selfie face detection (single face, centered, good quality)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.3
    ):
        """
        Initialize YuNet detector.

        Args:
            model_path: Path to yunet.onnx. If None, uses config.
            conf_threshold: Confidence threshold (0.0-1.0). Lower = more lenient (default: 0.4)
            nms_threshold: Non-maximum suppression threshold
        """
        if model_path is None:
            models_dir = Path(config.get("paths", "models_dir", default="models"))
            model_file = config.get("models", "face_detection", "local_file", default="yunet.onnx")
            model_path = str(models_dir / model_file)

        if not Path(model_path).exists():
            error_msg = (
                f"YuNet model not found at {model_path}. "
                f"Run: python scripts/download_models.py"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.detector = None
        self._last_size = (0, 0)
        self._lock = threading.Lock()  # Thread lock for detector access

        logger.info(f"YuNet initialized: {Path(model_path).name} (conf={conf_threshold}, nms={nms_threshold})")

    def _ensure_detector(self, width: int, height: int) -> None:
        """
        Initialize or resize detector for image size.
        Thread-safe implementation for concurrent requests with different image dimensions.
        Uses setInputSize() to dynamically adjust detector without recreation.
        """
        if self.detector is None:
            # First-time initialization with default size
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.model_path,
                    config="",
                    input_size=(640, 640),  # Default size, will be adjusted per image
                    score_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=5000
                )
                logger.info(f"YuNet detector created (supports dynamic resizing)")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
                raise
        
        # Always set input size for current image (thread-safe, no recreation needed)
        if self._last_size != (width, height):
            try:
                self.detector.setInputSize((width, height))
                self._last_size = (width, height)
                logger.debug(f"Detector resized to: {width}x{height}")
            except Exception as e:
                logger.error(f"Failed to resize detector: {e}")
                raise

    def detect(
        self,
        image: np.ndarray,
        return_largest: bool = True
    ) -> Optional[FaceDetectionResult]:
        """
        Detect face(s) in image.

        Args:
            image: Input image (BGR format)
            return_largest: If True, return only largest face. For KYC, should always be True.

        Returns:
            FaceDetectionResult or None if no face found
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect()")
            return None

        h, w = image.shape[:2]

        # Downscale large images before detection — YuNet struggles on very large images
        # (e.g. 3024x4032 full-res phone selfies) where the face occupies most of the frame.
        # Detection is run on the scaled image; bounding boxes are scaled back to original.
        MAX_DETECT_DIM = 1280
        scale = 1.0
        detect_image = image
        if max(h, w) > MAX_DETECT_DIM:
            scale = MAX_DETECT_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            detect_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Downscaled image for detection: {w}x{h} → {new_w}x{new_h} (scale={scale:.3f})")
            w, h = new_w, new_h

        # Use lock to ensure thread-safe access to detector (prevents OpenCV race conditions)
        with self._lock:
            self._ensure_detector(w, h)

            try:
                # Detect faces
                _, faces = self.detector.detect(detect_image)

                if faces is None or len(faces) == 0:
                    logger.debug(f"No faces detected (threshold={self.conf_threshold})")
                    return None

                # Parse detections - YuNet format: [x, y, w, h, 5 landmarks (x,y pairs), confidence]
                # If image was downscaled for detection, scale coordinates back to original size
                detections = []
                for face in faces:
                    bbox = face[:4].copy()
                    landmarks = face[4:14].reshape(5, 2).copy()
                    confidence = float(face[14])

                    if scale != 1.0:
                        bbox = (bbox / scale).astype(np.int32)
                        landmarks = (landmarks / scale).astype(np.int32)
                    else:
                        bbox = bbox.astype(np.int32)
                        landmarks = landmarks.astype(np.int32)

                    detections.append(FaceDetectionResult(
                        bbox=bbox,
                        confidence=confidence,
                        landmarks=landmarks
                    ))
                
                # Log all detections for debugging (always log to understand what's being detected)
                logger.info(f"YuNet detected {len(detections)} face(s):")
                for i, det in enumerate(detections):
                    face_area = det.bbox[2] * det.bbox[3]
                    logger.info(f"  Face {i+1}: size={det.bbox[2]}x{det.bbox[3]}, area={face_area}, conf={det.confidence:.3f}, pos=[{det.bbox[0]},{det.bbox[1]}]")

                if return_largest:
                    # Apply multiple filters to eliminate false positives:
                    # 1. Minimum size: 30x30 pixels (lowered to handle small ID document photos)
                    # 2. Confidence: tied to conf_threshold from config (no separate hardcoded value)
                    # 3. Aspect ratio: between 0.5 and 2.0 (faces are roughly square)
                    min_face_size = 30
                    min_confidence = self.conf_threshold
                    min_aspect_ratio = 0.5
                    max_aspect_ratio = 2.0
                    
                    valid_detections = []
                    for det in detections:
                        width, height = det.bbox[2], det.bbox[3]
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Check all criteria
                        size_ok = width >= min_face_size and height >= min_face_size
                        conf_ok = det.confidence >= min_confidence
                        aspect_ok = min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
                        
                        if size_ok and conf_ok and aspect_ok:
                            valid_detections.append(det)
                        else:
                            # Log why this detection was filtered out
                            reasons = []
                            if not size_ok:
                                reasons.append(f"size={width}x{height}<{min_face_size}")
                            if not conf_ok:
                                reasons.append(f"conf={det.confidence:.3f}<{min_confidence}")
                            if not aspect_ok:
                                reasons.append(f"aspect={aspect_ratio:.2f} out of range")
                            logger.debug(f"Filtered out detection: {', '.join(reasons)}")
                    
                    if not valid_detections:
                        # All detected faces failed quality filters (likely false positives)
                        if detections:
                            logger.warning(
                                f"Detected {len(detections)} face(s) but all failed quality filters. "
                                f"This usually means: (1) Face is too small/far from camera, "
                                f"(2) Poor image quality/blur, (3) False detections (logos, patterns). "
                                f"Required: size≥{min_face_size}x{min_face_size}, confidence≥{min_confidence:.2f}, aspect ratio {min_aspect_ratio}-{max_aspect_ratio}"
                            )
                        else:
                            logger.debug(f"No faces detected (threshold={self.conf_threshold})")
                        return None
                    
                    # Select largest valid face
                    largest = max(valid_detections, key=lambda x: x.bbox[2] * x.bbox[3])
                    
                    logger.info(f"Face detected: conf={largest.confidence:.3f}, bbox={largest.bbox.tolist()}, size={largest.bbox[2]}x{largest.bbox[3]}")
                    
                    # Log if we filtered out smaller detections
                    if len(valid_detections) < len(detections):
                        filtered_count = len(detections) - len(valid_detections)
                        logger.info(f"Filtered out {filtered_count} tiny false positive(s)")
                    
                    return largest
                
                # Sort by confidence
                detections.sort(key=lambda x: x.confidence, reverse=True)
                return detections[0] if detections else None
                
            except Exception as e:
                logger.error(f"Face detection failed: {e}")
                return None

    def extract_face(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        padding: float = 0.2,
        target_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        Crop and resize face from image.

        Args:
            image: Source image
            bbox: Bounding box [x, y, w, h]
            padding: Padding ratio around face (0.2 = 20% extra space)
            target_size: Output size (width, height). 112x112 is standard for InsightFace

        Returns:
            Cropped and resized face image
        """
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        # Ensure bounds
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        # Crop
        face_crop = image[y1:y2, x1:x2]

        # Resize to target size for embedding extraction
        if target_size is not None and face_crop.size > 0:
            face_crop = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

        return face_crop

    def detect_and_extract(
        self,
        image: np.ndarray,
        padding: float = 0.2,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[FaceDetectionResult]:
        """
        One-shot: detect largest face and extract crop.

        Args:
            image: Input image
            padding: Padding around detected face
            target_size: Resize extracted face to this size

        Returns:
            FaceDetectionResult with face_crop populated, or None
        """
        result = self.detect(image, return_largest=True)
        
        if result is None:
            return None

        # Extract face crop
        result.face_crop = self.extract_face(
            image,
            result.bbox,
            padding=padding,
            target_size=target_size
        )

        return result


# ============================================================================
# Singleton Pattern - Thread-safe
# ============================================================================

_detector_instance: Optional[YuNetFaceDetector] = None
_detector_lock = threading.Lock()


def get_face_detector() -> YuNetFaceDetector:
    """
    Thread-safe singleton getter.
    Ensures only one detector instance per process.
    """
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = YuNetFaceDetector(
                    conf_threshold=config.get("models", "face_detection", "conf_threshold", default=0.4),
                    nms_threshold=config.get("models", "face_detection", "nms_threshold", default=0.3)
                )
                logger.info("Face detector singleton created")
    return _detector_instance


def reset_detector() -> None:
    """Reset singleton (useful for testing)"""
    global _detector_instance
    _detector_instance = None
    logger.info("Face detector singleton reset")