# app/services/liveness_utils.py

"""
Utility functions for liveness detection.
Adapted from face_liveness_detection-Anti-spoofing/f_utils.py
Compatible with YuNet face detection format.
"""

import numpy as np
from typing import List, Tuple, Union
import cv2

from utils.logger import get_logger

logger = get_logger(__name__, log_file="liveness.log")


def get_areas(boxes: Union[List, np.ndarray]) -> List[float]:
    """
    Calculate area for each bounding box.
    
    Args:
        boxes: List/array of bounding boxes
               Format options:
               - [x, y, w, h] (YuNet format)
               - [x0, y0, x1, y1] (absolute coordinates)
    
    Returns:
        List of areas (floats)
    """
    areas = []
    for box in boxes:
        if len(box) == 4:
            if isinstance(box, np.ndarray):
                box = box.tolist()
            
            # Check format: [x, y, w, h] vs [x0, y0, x1, y1]
            # If w > image_width or h > image_height, it's likely absolute coords
            # Otherwise, assume it's [x, y, w, h]
            # Simple heuristic: if x1-x0 > 1000 or y1-y0 > 1000, likely absolute
            if len(box) == 4:
                x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                
                # Check if it's [x, y, w, h] format (YuNet)
                # If x1 and y1 are very large compared to x0/y0, might be absolute
                # Better: check if values look like coordinates vs dimensions
                if x1 < 10000 and y1 < 10000:
                    # Could be either format, try to detect
                    # If x1 < x0 or y1 < y0, it's definitely [x, y, w, h]
                    if x1 < x0:
                        # Format: [x, y, w, h]
                        area = x1 * y1  # w * h
                    else:
                        # Format: [x0, y0, x1, y1]
                        area = (x1 - x0) * (y1 - y0)
                else:
                    # Likely absolute coordinates (very large values)
                    area = (x1 - x0) * (y1 - y0)
            else:
                area = 0
        else:
            area = 0
        
        areas.append(float(area))
    
    return areas


def convert_bbox_to_absolute(bbox: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert YuNet bbox format [x, y, w, h] to absolute coordinates [x0, y0, x1, y1].
    
    Args:
        bbox: Bounding box in [x, y, w, h] format
        image_shape: (height, width) of the image
    
    Returns:
        Bounding box in [x0, y0, x1, y1] format
    """
    if bbox is None or len(bbox) < 4:
        return np.array([])
    
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Convert to absolute coordinates
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(image_shape[1], int(x + w))
    y1 = min(image_shape[0], int(y + h))
    
    return np.array([x0, y0, x1, y1], dtype=np.int32)


def convert_absolute_to_yunet(bbox: np.ndarray) -> np.ndarray:
    """
    Convert absolute bbox [x0, y0, x1, y1] to YuNet format [x, y, w, h].
    
    Args:
        bbox: Bounding box in [x0, y0, x1, y1] format
    
    Returns:
        Bounding box in [x, y, w, h] format
    """
    if bbox is None or len(bbox) < 4:
        return np.array([])
    
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    x = x0
    y = y0
    w = x1 - x0
    h = y1 - y0
    
    return np.array([x, y, w, h], dtype=np.int32)


def convert_rectangles2array(rectangles, image: np.ndarray) -> np.ndarray:
    """
    Convert dlib rectangles to numpy array format.
    Adapted from original f_utils.py for compatibility.
    
    Args:
        rectangles: dlib rectangle objects or list of bboxes
        image: Image array for dimension checking
    
    Returns:
        Array of bounding boxes in [x0, y0, x1, y1] format
    """
    res = np.array([])
    
    if not rectangles or len(rectangles) == 0:
        return res
    
    h, w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
    
    for box in rectangles:
        try:
            # Handle dlib rectangle objects
            if hasattr(box, 'left') and hasattr(box, 'top'):
                # dlib rectangle
                x0 = max(0, box.left())
                y0 = max(0, box.top())
                x1 = min(w, box.right())
                y1 = min(h, box.bottom())
            elif isinstance(box, (list, tuple, np.ndarray)):
                # Already in array format
                if len(box) == 4:
                    if isinstance(box, np.ndarray):
                        box = box.tolist()
                    
                    # Check if [x, y, w, h] or [x0, y0, x1, y1]
                    x0, y0, x1_or_w, y1_or_h = box
                    
                    # Heuristic: if x1_or_w < x0, it's width, not x1
                    if x1_or_w < x0:
                        # Format: [x, y, w, h] - convert to absolute
                        x0 = max(0, int(x0))
                        y0 = max(0, int(y0))
                        x1 = min(w, int(x0 + x1_or_w))
                        y1 = min(h, int(y0 + y1_or_h))
                    else:
                        # Format: [x0, y0, x1, y1] - already absolute
                        x0 = max(0, int(x0))
                        y0 = max(0, int(y0))
                        x1 = min(w, int(x1_or_w))
                        y1 = min(h, int(y1_or_h))
                else:
                    continue
            else:
                continue
            
            new_box = np.array([x0, y0, x1, y1], dtype=np.int32)
            
            if res.size == 0:
                res = np.expand_dims(new_box, axis=0)
            else:
                res = np.vstack((res, new_box))
        
        except Exception as e:
            logger.warning(f"Error converting box: {e}")
            continue
    
    return res


def get_largest_face(bboxes: Union[List, np.ndarray]) -> int:
    """
    Find index of largest face from bounding boxes.
    
    Args:
        bboxes: List/array of bounding boxes
    
    Returns:
        Index of largest face, or -1 if empty
    """
    if not bboxes or len(bboxes) == 0:
        return -1
    
    areas = get_areas(bboxes)
    if not areas:
        return -1
    
    return int(np.argmax(areas))


def extract_face_roi(image: np.ndarray, bbox: np.ndarray, padding: float = 0.1) -> np.ndarray:
    """
    Extract face region of interest from image.
    
    Args:
        image: Input image
        bbox: Bounding box in [x, y, w, h] or [x0, y0, x1, y1] format
        padding: Padding ratio around face (0.1 = 10% padding)
    
    Returns:
        Extracted face image, or None if extraction fails
    """
    if image is None or bbox is None or len(bbox) < 4:
        return None
    
    h, w = image.shape[:2]
    
    # Convert bbox to absolute if needed
    if len(bbox) == 4:
        # Check if [x, y, w, h] or [x0, y0, x1, y1]
        x0, y0, x1_or_w, y1_or_h = bbox[0], bbox[1], bbox[2], bbox[3]
        
        if x1_or_w < x0 or y1_or_h < y0:
            # Format: [x, y, w, h]
            x0 = max(0, int(x0 - padding * x1_or_w))
            y0 = max(0, int(y0 - padding * y1_or_h))
            x1 = min(w, int(x0 + x1_or_w + 2 * padding * x1_or_w))
            y1 = min(h, int(y0 + y1_or_h + 2 * padding * y1_or_h))
        else:
            # Format: [x0, y0, x1, y1]
            width = x1_or_w - x0
            height = y1_or_h - y0
            x0 = max(0, int(x0 - padding * width))
            y0 = max(0, int(y0 - padding * height))
            x1 = min(w, int(x1_or_w + padding * width))
            y1 = min(h, int(y1_or_h + padding * height))
    
    try:
        face_roi = image[y0:y1, x0:x1]
        return face_roi if face_roi.size > 0 else None
    except Exception as e:
        logger.warning(f"Face ROI extraction failed: {e}")
        return None


def validate_bbox(bbox: np.ndarray, image_shape: Tuple[int, int]) -> bool:
    """
    Validate bounding box is within image bounds.
    
    Args:
        bbox: Bounding box
        image_shape: (height, width) of image
    
    Returns:
        True if valid, False otherwise
    """
    if bbox is None or len(bbox) < 4:
        return False
    
    h, w = image_shape
    
    # Convert to absolute if needed
    if len(bbox) == 4:
        x0, y0, x1_or_w, y1_or_h = bbox[0], bbox[1], bbox[2], bbox[3]
        
        if x1_or_w < x0:
            # Format: [x, y, w, h]
            x0 = int(x0)
            y0 = int(y0)
            x1 = int(x0 + x1_or_w)
            y1 = int(y0 + y1_or_h)
        else:
            # Format: [x0, y0, x1, y1]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1_or_w), int(y1_or_h)
        
        # Check bounds
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return False
        if x0 >= x1 or y0 >= y1:
            return False
        
        return True
    
    return False

