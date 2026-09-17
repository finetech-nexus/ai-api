# app/services/profile_detector.py

"""
Face Orientation/Profile Detection using Haar Cascades.
Adapted from face_liveness_detection-Anti-spoofing/profile_detection/f_detector.py
Detects left and right face orientations for liveness challenges.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import threading

from configs.config import config
from utils.logger import get_logger
from app.services import liveness_utils

logger = get_logger(__name__, log_file="liveness.log")


class ProfileDetector:
    """
    Face orientation detector using Haar cascades.
    Detects left and right profile faces for liveness challenges.
    """
    
    def __init__(
        self,
        frontal_cascade_path: Optional[str] = None,
        profile_cascade_path: Optional[str] = None
    ):
        """
        Initialize Haar cascade classifiers.
        
        Args:
            frontal_cascade_path: Path to frontal face cascade (optional, uses config if None)
            profile_cascade_path: Path to profile face cascade (optional, uses config if None)
        """
        # Get paths from config
        if frontal_cascade_path is None:
            models_dir = Path(config.get("paths", "models_dir", default="models"))
            frontal_file = config.get("liveness", "models", "haar_frontal", default="haarcascade_frontalface_alt.xml")
            frontal_cascade_path = str(models_dir / frontal_file)
        
        if profile_cascade_path is None:
            models_dir = Path(config.get("paths", "models_dir", default="models"))
            profile_file = config.get("liveness", "models", "haar_profile", default="haarcascade_profileface.xml")
            profile_cascade_path = str(models_dir / profile_file)
        
        # Load cascade classifiers
        self.frontal_cascade = cv2.CascadeClassifier(frontal_cascade_path)
        if self.frontal_cascade.empty():
            raise FileNotFoundError(f"Failed to load frontal cascade: {frontal_cascade_path}")
        
        self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
        if self.profile_cascade.empty():
            raise FileNotFoundError(f"Failed to load profile cascade: {profile_cascade_path}")
        
        logger.info(f"ProfileDetector initialized")
        logger.info(f"  Frontal cascade: {frontal_cascade_path}")
        logger.info(f"  Profile cascade: {profile_cascade_path}")
    
    def _detect_with_cascade(
        self,
        gray_image: np.ndarray,
        cascade: cv2.CascadeClassifier
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces using Haar cascade.
        
        Args:
            gray_image: Grayscale image
            cascade: Haar cascade classifier
        
        Returns:
            Tuple of (rectangles, confidence_scores)
            Rectangles are in [x, y, w, h] format
        """
        try:
            # Use detectMultiScale3 to get confidence scores
            rects, _, confidence = cascade.detectMultiScale3(
                gray_image,
                scaleFactor=1.3,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
                outputRejectLevels=True
            )
            
            if len(rects) == 0:
                return np.array([]), np.array([])
            
            # Convert [x, y, w, h] to [x0, y0, x1, y1] format for consistency
            rects_absolute = rects.copy()
            rects_absolute[:, 2] += rects_absolute[:, 0]  # x1 = x0 + w
            rects_absolute[:, 3] += rects_absolute[:, 1]  # y1 = y0 + h
            
            return rects_absolute, confidence
        
        except Exception as e:
            logger.warning(f"Cascade detection failed: {e}")
            return np.array([]), np.array([])
    
    def _convert_right_box(
        self,
        gray_image: np.ndarray,
        box_right: np.ndarray
    ) -> np.ndarray:
        """
        Convert right profile box coordinates back after flipping.
        
        When we detect right profile, we flip the image first.
        This function converts the detected box coordinates back to original image coordinates.
        
        Args:
            gray_image: Original grayscale image
            box_right: Detected box in flipped image coordinates [x0, y0, x1, y1]
        
        Returns:
            Box in original image coordinates [x0, y0, x1, y1]
        """
        if len(box_right) == 0:
            return np.array([])
        
        x_max = gray_image.shape[1]
        
        res = np.array([])
        for box in box_right:
            # Convert from flipped coordinates back to original
            # In flipped image: x is measured from right edge
            # In original: x = image_width - flipped_x
            converted_box = box.copy()
            converted_box[0] = x_max - box[2]  # x0_orig = width - x1_flipped
            converted_box[2] = x_max - box[0]  # x1_orig = width - x0_flipped
            # y coordinates stay the same
            
            if res.size == 0:
                res = np.expand_dims(converted_box, axis=0)
            else:
                res = np.vstack((res, converted_box))
        
        return res
    
    def detect_orientation(
        self,
        image: np.ndarray
    ) -> Tuple[List, List[str]]:
        """
        Detect face orientation (left or right profile).
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Tuple of (boxes, orientations)
            - boxes: List of bounding boxes [[x0, y0, x1, y1], ...]
            - orientations: List of orientation strings ['left'] or ['right'] or []
        """
        if image is None or image.size == 0:
            return [], []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect left profile (direct detection)
        box_left, confidence_left = self._detect_with_cascade(gray, self.profile_cascade)
        
        left_boxes = []
        left_names = []
        if len(box_left) > 0:
            left_boxes = box_left.tolist() if isinstance(box_left, np.ndarray) else box_left
            left_names = ["left"] * len(left_boxes)
        
        # Detect right profile (by flipping image)
        gray_flipped = cv2.flip(gray, 1)  # Flip horizontally
        box_right_flipped, confidence_right = self._detect_with_cascade(gray_flipped, self.profile_cascade)
        
        right_boxes = []
        right_names = []
        if len(box_right_flipped) > 0:
            # Convert coordinates back to original image
            box_right = self._convert_right_box(gray, box_right_flipped)
            if len(box_right) > 0:
                right_boxes = box_right.tolist() if isinstance(box_right, np.ndarray) else box_right
                right_names = ["right"] * len(right_boxes)
        
        # Combine all detections
        all_boxes = left_boxes + right_boxes
        all_names = left_names + right_names
        
        if len(all_boxes) == 0:
            return [], []
        
        # Return largest face (by area)
        if len(all_boxes) == 1:
            return [all_boxes[0]], [all_names[0]]
        
        # Find largest by area
        areas = liveness_utils.get_areas(all_boxes)
        largest_index = int(np.argmax(areas))
        
        return [all_boxes[largest_index]], [all_names[largest_index]]
    
    def detect_orientation_frame(
        self,
        image: np.ndarray
    ) -> Dict[str, any]:
        """
        Detect face orientation in a single frame.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary with:
            - orientation: 'left', 'right', or None
            - box: Bounding box [x0, y0, x1, y1] or None
            - confidence: Detection confidence (if available)
        """
        boxes, orientations = self.detect_orientation(image)
        
        if len(boxes) == 0 or len(orientations) == 0:
            return {
                "orientation": None,
                "box": None,
                "confidence": 0.0
            }
        
        return {
            "orientation": orientations[0],
            "box": boxes[0],
            "confidence": 1.0  # Haar cascades don't provide confidence, use 1.0 if detected
        }
    
    def detect_orientation_batch(
        self,
        frames: List[np.ndarray]
    ) -> Dict[str, any]:
        """
        Detect face orientations across a batch of frames.
        
        Args:
            frames: List of image frames (BGR format)
        
        Returns:
            Dictionary with:
            - orientations: List of detected orientations per frame
            - orientation_frames: Frame indices where orientations were detected
            - left_frames: Frame indices where left was detected
            - right_frames: Frame indices where right was detected
            - face_detection_ratio: Ratio of frames with any orientation detected
        """
        orientations_list = []
        left_frames = []
        right_frames = []
        
        for i, frame in enumerate(frames):
            result = self.detect_orientation_frame(frame)
            orientation = result["orientation"]
            orientations_list.append(orientation)
            
            if orientation == "left":
                left_frames.append(i)
            elif orientation == "right":
                right_frames.append(i)
        
        frames_with_orientation = len([o for o in orientations_list if o is not None])
        face_detection_ratio = frames_with_orientation / len(frames) if len(frames) > 0 else 0.0
        
        return {
            "orientations": orientations_list,
            "orientation_frames": left_frames + right_frames,
            "left_frames": left_frames,
            "right_frames": right_frames,
            "face_detection_ratio": face_detection_ratio,
            "total_detections": frames_with_orientation
        }


# ============================================================================
# Singleton Pattern - Thread-safe
# ============================================================================

_detector_instance: Optional[ProfileDetector] = None
_detector_lock = threading.Lock()


def get_profile_detector() -> ProfileDetector:
    """
    Get or create singleton ProfileDetector instance.
    Thread-safe.
    """
    global _detector_instance
    
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = ProfileDetector()
                logger.info("ProfileDetector singleton created")
    
    return _detector_instance

