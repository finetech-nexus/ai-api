# app/services/liveness_detector.py

"""
Main Liveness Detection Orchestrator.
Combines blink detection, profile detection, and challenge validation.
Adapted from face_liveness_detection-Anti-spoofing/f_liveness_detection.py
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
from collections import Counter

from app.services.blink_detector import get_blink_detector, BlinkDetector
from app.services.profile_detector import get_profile_detector, ProfileDetector
from app.services.liveness_challenges import ChallengeGenerator, ChallengeType, ChallengeStatus
from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="liveness.log")


class LivenessDetectionResult:
    """Encapsulates liveness detection results."""
    
    def __init__(
        self,
        blinks: int = 0,
        orientation: Optional[str] = None,
        orientation_box: Optional[List] = None,
        face_detected: bool = False,
        face_box: Optional[List] = None,
        ear_value: float = 1.0,
        is_blinking: bool = False
    ):
        self.blinks = blinks
        self.orientation = orientation  # 'left', 'right', or None
        self.orientation_box = orientation_box
        self.face_detected = face_detected
        self.face_box = face_box
        self.ear_value = ear_value  # Eye Aspect Ratio
        self.is_blinking = is_blinking
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "blinks": self.blinks,
            "orientation": self.orientation,
            "orientation_box": self.orientation_box,
            "face_detected": self.face_detected,
            "face_box": self.face_box,
            "ear_value": round(self.ear_value, 4),
            "is_blinking": self.is_blinking
        }
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy format (compatible with original repo)."""
        return {
            "box_face_frontal": [self.face_box] if self.face_box else [],
            "box_orientation": [self.orientation_box] if self.orientation_box else [],
            "emotion": [],  # Skipped for MVP
            "orientation": [self.orientation] if self.orientation else [],
            "total_blinks": self.blinks,
            "count_blinks_consecutives": 1 if self.is_blinking else 0
        }


class LivenessDetector:
    """
    Main liveness detection orchestrator.
    Combines blink detection, profile detection, and challenge validation.
    """
    
    def __init__(
        self,
        blink_detector: Optional[BlinkDetector] = None,
        profile_detector: Optional[ProfileDetector] = None,
        challenge_generator: Optional[ChallengeGenerator] = None
    ):
        """
        Initialize liveness detector with component detectors.
        
        Args:
            blink_detector: BlinkDetector instance (uses singleton if None)
            profile_detector: ProfileDetector instance (uses singleton if None)
            challenge_generator: ChallengeGenerator instance (uses singleton if None)
        """
        from app.services.liveness_challenges import get_challenge_generator
        self.blink_detector = blink_detector or get_blink_detector()
        self.profile_detector = profile_detector or get_profile_detector()
        self.challenge_generator = challenge_generator or get_challenge_generator()
        
        logger.info("LivenessDetector initialized")
    
    def detect_frame(
        self,
        image: np.ndarray,
        counter: int = 0,
        total: int = 0
    ) -> Tuple[LivenessDetectionResult, int, int]:
        """
        Detect liveness in a single frame.
        
        Args:
            image: Input image (BGR format)
            counter: Current consecutive blink frames counter
            total: Total blink count
        
        Returns:
            Tuple of (LivenessDetectionResult, updated_counter, updated_total)
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_frame")
            return LivenessDetectionResult(), counter, total
        
        # Convert to grayscale for profile detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Blink detection (works on BGR image)
        counter, total, ear, is_blinking = self.blink_detector.detect_blink_frame(image, counter, total)
        
        # 2. Profile/Orientation detection (works on grayscale)
        orientation_result = self.profile_detector.detect_orientation_frame(image)
        
        # 3. Build result object
        result = LivenessDetectionResult(
            blinks=total,
            orientation=orientation_result.get("orientation"),
            orientation_box=orientation_result.get("box"),
            face_detected=ear < 1.0,  # EAR < 1.0 means face was detected
            face_box=None,  # MediaPipe doesn't provide face box, only landmarks
            ear_value=ear,
            is_blinking=is_blinking
        )
        
        return result, counter, total
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        initial_counter: int = 0,
        initial_total: int = 0
    ) -> Dict[str, Any]:
        """
        Detect liveness across a batch of frames.
        
        Args:
            frames: List of image frames (BGR format)
            initial_counter: Starting consecutive blink frames counter
            initial_total: Starting total blink count
        
        Returns:
            Dictionary with comprehensive liveness results
        """
        if not frames:
            return {
                "total_blinks": 0,
                "orientations": [],
                "face_detection_ratio": 0.0,
                "results": []
            }
        
        results_list = []
        counter = initial_counter
        total = initial_total
        
        # Process each frame
        for i, frame in enumerate(frames):
            result, counter, total = self.detect_frame(frame, counter, total)
            results_list.append(result.to_dict())
        
        # Get batch orientation detection results
        orientation_batch = self.profile_detector.detect_orientation_batch(frames)
        
        # Aggregate results
        new_blinks = total - initial_total
        orientations_detected = [r.get("orientation") for r in results_list if r.get("orientation")]
        face_detected_count = sum(1 for r in results_list if r.get("face_detected"))
        face_detection_ratio = face_detected_count / len(frames) if len(frames) > 0 else 0.0
        
        return {
            "total_blinks": new_blinks,
            "final_blink_count": total,
            "orientations": [r.get("orientation") for r in results_list],
            "left_orientations": orientation_batch.get("left_frames", []),
            "right_orientations": orientation_batch.get("right_frames", []),
            "face_detection_ratio": face_detection_ratio,
            "results": results_list,
            "frame_count": len(frames)
        }
    
    def verify_challenge(
        self,
        challenge_id: str,
        frames: List[np.ndarray],
        initial_counter: int = 0,
        initial_total: int = 0
    ) -> Tuple[ChallengeStatus, str, Dict[str, Any]]:
        """
        Verify a liveness challenge with frame sequence.
        
        Args:
            challenge_id: Challenge ID to verify
            frames: List of frames captured during challenge
            initial_counter: Starting blink counter
            initial_total: Starting blink total
        
        Returns:
            Tuple of (status, message, detection_results)
        """
        if not frames:
            status, message = ChallengeStatus.FAIL, "No frames provided"
            return status, message, {}
        
        # First validate challenge exists and is not expired
        is_valid, challenge = self.challenge_generator.validate_challenge(challenge_id)
        if not is_valid or challenge is None:
            logger.warning(f"Challenge validation failed for {challenge_id}: is_valid={is_valid}")
            status, message = ChallengeStatus.INVALID, "Challenge not found or expired. Please generate a new challenge."
            return status, message, {}
        
        # Detect liveness in frames
        batch_results = self.detect_batch(frames, initial_counter, initial_total)
        
        # Extract detection results for validation
        # Include both individual orientation and full orientations list for multi-challenge
        detection_results = {
            "blinks": batch_results.get("total_blinks", 0),
            "orientation": self._get_primary_orientation(batch_results.get("orientations", [])),
            "orientations": batch_results.get("orientations", []),  # Full list for multi-challenge
            "face_detected": batch_results.get("face_detection_ratio", 0.0) > 0.5
        }
        
        # Validate challenge response
        status, message = self.challenge_generator.validate_response(challenge_id, detection_results)
        
        return status, message, {
            "detection_results": detection_results,
            "batch_results": batch_results
        }
    
    def _get_primary_orientation(self, orientations: List[Optional[str]]) -> Optional[str]:
        """
        Get primary orientation from list (most common).
        
        Args:
            orientations: List of detected orientations
        
        Returns:
            Most common orientation or None
        """
        if not orientations:
            return None
        
        # Filter out None values
        valid_orientations = [o for o in orientations if o is not None]
        
        if not valid_orientations:
            return None
        
        # Count occurrences
        counts = Counter(valid_orientations)
        return counts.most_common(1)[0][0]
    
    def detect_liveness(
        self,
        image: np.ndarray,
        counter: int = 0,
        total: int = 0
    ) -> Dict[str, Any]:
        """
        Detect liveness (legacy compatibility function).
        Compatible with original repo's detect_liveness signature.
        
        Args:
            image: Input image (BGR format)
            counter: Current consecutive blink frames counter
            total: Total blink count
        
        Returns:
            Dictionary in legacy format
        """
        result, counter, total = self.detect_frame(image, counter, total)
        legacy_dict = result.to_legacy_dict()
        legacy_dict["total_blinks"] = total
        legacy_dict["count_blinks_consecutives"] = counter
        return legacy_dict


# ============================================================================
# Singleton Pattern
# ============================================================================

_detector_instance: Optional[LivenessDetector] = None
_detector_lock = threading.Lock()


def get_liveness_detector() -> LivenessDetector:
    """
    Get or create singleton LivenessDetector instance.
    Thread-safe.
    """
    global _detector_instance
    
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = LivenessDetector()
                logger.info("LivenessDetector singleton created")
    
    return _detector_instance

