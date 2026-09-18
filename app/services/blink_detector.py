# app/services/blink_detector.py

"""
Blink Detection using MediaPipe FaceMesh.
Adapted from face_liveness_detection-Anti-spoofing/blink_detection/f_blink_detection.py
Replaced dlib with MediaPipe for better Railway deployment compatibility.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.spatial import distance as dist
import threading
import mediapipe as mp
from mediapipe.python.solutions import face_mesh

from configs.config import config
from utils.logger import get_logger

logger = get_logger(__name__, log_file="liveness.log")


class BlinkDetector:
    """
    Blink detection using MediaPipe FaceMesh.
    Uses EAR (Eye Aspect Ratio) calculation similar to dlib approach.
    """
    
    # MediaPipe FaceMesh eye landmark indices (468-point model)
    # Eye landmarks ordered for EAR calculation (matching dlib structure)
    # Order: [left_corner, top_near_left, top_near_right, right_corner, bottom_near_right, bottom_near_left]
    # Left eye (from camera perspective)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # [outer, top_left, top_right, inner, bottom_inner, bottom_outer]
    # Right eye (from camera perspective)  
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # [outer, top_left, top_right, inner, bottom_inner, bottom_outer]
    
    def __init__(
        self,
        ear_threshold: Optional[float] = None,
        consecutive_frames: Optional[int] = None,
        min_detection_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe FaceMesh for blink detection.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold for blink detection (default from config)
            consecutive_frames: Frames below threshold to count as blink (default from config)
            min_detection_confidence: Minimum confidence for face detection
        """
        # Get thresholds from config
        if ear_threshold is None:
            ear_threshold = config.get("liveness", "blink", "ear_threshold", default=0.23)
        
        if consecutive_frames is None:
            consecutive_frames = config.get("liveness", "blink", "consecutive_frames", default=1)
        
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # Only detect one face for liveness
            refine_landmarks=True,  # Get more accurate landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        logger.info(f"BlinkDetector initialized (EAR threshold: {ear_threshold}, consecutive: {consecutive_frames})")
    
    def eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (vertical_distances) / (2 * horizontal_distance)
        
        Args:
            eye_landmarks: Array of 6 eye landmark points [(x, y), ...]
                           Order: [outer_corner, top, inner_corner, bottom_mid, bottom_outer, top_outer]
        
        Returns:
            EAR value (lower = more closed, typically 0.2-0.4)
        """
        if eye_landmarks is None or len(eye_landmarks) < 6:
            return 1.0  # Return high value (eyes open) if insufficient points
        
        try:
            # Extract the 6 eye points
            eye_points = eye_landmarks[:6]
            
            # Calculate vertical distances using standard EAR formula
            # Points order: [outer_corner(0), top_left(1), top_right(2), inner_corner(3), bottom_inner(4), bottom_outer(5)]
            # Vertical distance 1: top-left to bottom-left (point 1 to point 5)
            A = dist.euclidean(eye_points[1], eye_points[5])
            # Vertical distance 2: top-right to bottom-right (point 2 to point 4)
            B = dist.euclidean(eye_points[2], eye_points[4])
            
            # Horizontal distance: outer corner to inner corner (point 0 to point 3)
            C = dist.euclidean(eye_points[0], eye_points[3])
            
            # Avoid division by zero
            if C == 0:
                return 1.0
            
            # Calculate EAR
            ear = (A + B) / (2.0 * C)
            
            # Log for debugging if EAR is suspicious
            if ear < 0.15:  # Very closed eye
                logger.debug(f"Very low EAR detected: {ear:.3f}")
            
            return ear
        
        except Exception as e:
            logger.warning(f"EAR calculation failed: {e}")
            return 1.0  # Default to eyes open
    
    def extract_eye_landmarks(
        self,
        landmarks,
        image_width: int,
        image_height: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract eye landmarks from MediaPipe FaceMesh results.
        
        Args:
            landmarks: MediaPipe landmark list
            image_width: Image width (for coordinate conversion)
            image_height: Image height (for coordinate conversion)
        
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks) as numpy arrays
            Each array contains 6 points in pixel coordinates
        """
        if not landmarks:
            return None, None
        
        try:
            # MediaPipe landmarks are normalized (0-1), convert to pixel coordinates
            left_eye_points = []
            right_eye_points = []
            
            for idx in self.LEFT_EYE_INDICES:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    # Convert normalized to pixel coordinates
                    x = landmark.x * image_width
                    y = landmark.y * image_height
                    left_eye_points.append([x, y])
            
            for idx in self.RIGHT_EYE_INDICES:
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    # Convert normalized to pixel coordinates
                    x = landmark.x * image_width
                    y = landmark.y * image_height
                    right_eye_points.append([x, y])
            
            if len(left_eye_points) == 6 and len(right_eye_points) == 6:
                return np.array(left_eye_points), np.array(right_eye_points)
            else:
                logger.warning(f"Incomplete eye landmarks: left={len(left_eye_points)}, right={len(right_eye_points)}")
                return None, None
        
        except Exception as e:
            logger.warning(f"Eye landmark extraction failed: {e}")
            return None, None
    
    def detect_blink_frame(
        self,
        image: np.ndarray,
        counter: int = 0,
        total: int = 0
    ) -> Tuple[int, int, float, bool]:
        """
        Detect blink in a single frame.
        
        Args:
            image: Input image (BGR format)
            counter: Current consecutive frames below threshold
            total: Total blink count so far
        
        Returns:
            Tuple of (counter, total, ear, is_blinking)
            - counter: Updated consecutive frames counter
            - total: Updated total blink count
            - ear: Current Eye Aspect Ratio
            - is_blinking: True if eyes are currently closed/blinking
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect_blink_frame")
            return counter, total, 1.0, False
        
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]
            
            # Process with MediaPipe FaceMesh
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                # No face detected
                logger.debug("No face detected in frame")
                return counter, total, 1.0, False
            
            # Get first face (we only detect one)
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract eye landmarks
            left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, w, h)
            
            if left_eye is None or right_eye is None:
                logger.debug("Could not extract eye landmarks")
                return counter, total, 1.0, False
            
            # Calculate EAR for both eyes
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            
            # Use average EAR for detection (more stable than minimum)
            # But also check if either eye is blinking (more sensitive)
            avg_ear = (left_ear + right_ear) / 2.0
            min_ear = min(left_ear, right_ear)
            
            # Detect blink: either average is below threshold OR minimum is significantly below
            # This catches both synchronized and unsynchronized blinks
            is_blinking = avg_ear < self.ear_threshold or min_ear < (self.ear_threshold * 0.8)
            
            # Debug logging for blink detection (only log when blinking or close to threshold)
            if is_blinking or avg_ear < 0.3:
                logger.debug(f"EAR - Left: {left_ear:.3f}, Right: {right_ear:.3f}, Avg: {avg_ear:.3f}, Min: {min_ear:.3f}, Threshold: {self.ear_threshold}, Blinking: {is_blinking}")
            
            if is_blinking:
                counter += 1
                logger.debug(f"Eyes closed - Counter: {counter}/{self.consecutive_frames}, Avg EAR: {avg_ear:.3f}, Min EAR: {min_ear:.3f}")
            else:
                # If eyes were closed for sufficient frames, count as blink
                if counter >= self.consecutive_frames:
                    total += 1
                    logger.info(f"✅ Blink detected! Total: {total}, Avg EAR during blink: {avg_ear:.3f}, Counter was: {counter}")
                # Reset counter
                counter = 0
            
            return counter, total, avg_ear, is_blinking  # Return avg_ear for display
        
        except Exception as e:
            logger.error(f"Blink detection failed: {e}", exc_info=True)
            return counter, total, 1.0, False
    
    def detect_blinks_batch(
        self,
        frames: list,
        initial_counter: int = 0,
        initial_total: int = 0
    ) -> Dict[str, any]:
        """
        Detect blinks across a batch of frames.
        
        Args:
            frames: List of image frames (BGR format)
            initial_counter: Starting consecutive frames counter
            initial_total: Starting total blink count
        
        Returns:
            Dictionary with:
            - total_blinks: Total number of blinks detected
            - blink_frames: List of frame indices where blinks were detected
            - ear_values: List of EAR values per frame
            - face_detected_ratio: Ratio of frames where face was detected
        """
        counter = initial_counter
        total = initial_total
        blink_frames = []
        ear_values = []
        frames_with_face = 0
        
        for i, frame in enumerate(frames):
            counter, total, ear, is_blinking = self.detect_blink_frame(frame, counter, total)
            
            ear_values.append(ear)
            
            if ear < 1.0:  # Face detected (ear would be 1.0 if no face)
                frames_with_face += 1
            
            # Check if this frame completed a blink
            if counter == 0 and total > initial_total:
                blink_frames.append(i)
        
        face_detection_ratio = frames_with_face / len(frames) if len(frames) > 0 else 0.0
        
        result = {
            "total_blinks": total - initial_total,  # New blinks detected
            "blink_frames": blink_frames,
            "ear_values": ear_values,
            "face_detection_ratio": face_detection_ratio,
            "final_counter": counter,
            "final_total": total
        }
        
        logger.info(f"Batch blink detection: {total - initial_total} blinks in {len(frames)} frames")
        return result
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            logger.info("BlinkDetector cleaned up")


# ============================================================================
# Singleton Pattern - Thread-safe
# ============================================================================

_detector_instance: Optional[BlinkDetector] = None
_detector_lock = threading.Lock()


def get_blink_detector() -> BlinkDetector:
    """
    Get or create singleton BlinkDetector instance.
    Thread-safe.
    """
    global _detector_instance
    
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = BlinkDetector()
                logger.info("BlinkDetector singleton created")
    
    return _detector_instance

