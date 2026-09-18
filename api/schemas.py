# api/schemas.py

"""
Pydantic schemas for KYC API - FIXED for Ballerine frontend compatibility.
Added confidence_score field as required by frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class VerificationStatus(str, Enum):
    """Verification status for KYC workflow."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    ERROR = "error"


# ============================================================================
# OCR Response Models
# ============================================================================

class OCRFields(BaseModel):
    """Structured fields extracted from ID document."""
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    document_number: Optional[str] = None
    nationality: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    place_of_birth: Optional[str] = None
    address: Optional[str] = None
    gender: Optional[str] = None


class OCRData(BaseModel):
    """OCR extraction result."""
    document_type: str = Field(description="Detected document type")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall OCR confidence")
    extracted_text: str = Field(description="Full raw text extracted")
    fields: OCRFields = Field(description="Structured extracted fields")


# ============================================================================
# Face Verification Response Models
# ============================================================================

class SimilarityMetrics(BaseModel):
    """Face similarity metrics."""
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    euclidean_distance: float = Field(ge=0.0)


class FaceMatchData(BaseModel):
    """Face matching verification result."""
    verified: bool = Field(description="Whether faces match")
    confidence: float = Field(ge=0.0, le=1.0, description="Match confidence score")
    similarity_metrics: SimilarityMetrics
    threshold_used: float
    message: str


# ============================================================================
# Main KYC Verification Response
# ============================================================================

class KYCVerificationResponse(BaseModel):
    """
    Main KYC verification response - FIXED for Ballerine frontend.
    Added confidence_score as top-level field (required by frontend).
    """
    verification_status: VerificationStatus = Field(
        description="Overall verification status"
    )
    
    # ✅ FIX: Added confidence_score for frontend compatibility
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence (weighted: 60% face + 40% OCR)"
    )
    
    face_match_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Face matching confidence (0-1)"
    )
    
    ocr_data: OCRData = Field(description="OCR extraction results")
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional detailed breakdown
    face_verification_details: Optional[FaceMatchData] = None
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "verification_status": "approved",
                "confidence_score": 0.89,
                "face_match_score": 0.87,
                "ocr_data": {
                    "document_type": "national_id",
                    "confidence": 0.92,
                    "extracted_text": "Full name: John Doe | DOB: 01/01/1990",
                    "fields": {
                        "full_name": "John Doe",
                        "date_of_birth": "01/01/1990",
                        "document_number": "ABC123456",
                        "nationality": "German"
                    }
                },
                "processing_time_ms": 2500,
                "timestamp": "2024-10-11T10:30:00Z",
                "face_verification_details": {
                    "verified": True,
                    "confidence": 0.87,
                    "similarity_metrics": {
                        "cosine_similarity": 0.85,
                        "euclidean_distance": 0.42
                    },
                    "threshold_used": 0.4,
                    "message": "Faces match (85.0% similarity)"
                }
            }
        }


# ============================================================================
# OCR-Only Response
# ============================================================================

class OCROnlyResponse(BaseModel):
    """Response for OCR-only endpoint."""
    ocr_data: OCRData
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "ocr_data": {
                    "document_type": "drivers_license",
                    "confidence": 0.88,
                    "extracted_text": "Name: Jane Smith | DOB: 15/03/1985",
                    "fields": {
                        "full_name": "Jane Smith",
                        "date_of_birth": "15/03/1985"
                    }
                },
                "processing_time_ms": 3200,
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }


# ============================================================================
# Error Response
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "No face detected in ID document",
                "details": {"confidence_threshold": 0.6},
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }


# ============================================================================
# Health Check Response
# ============================================================================

class ModelStatus(BaseModel):
    """Individual model status."""
    loaded: bool
    name: str
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response with model status."""
    status: str = Field(description="Overall service status: healthy|degraded|unhealthy")
    version: str
    models: Dict[str, ModelStatus]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models": {
                    "face_detector": {"loaded": True, "name": "yunet", "error": None},
                    "face_matcher": {"loaded": True, "name": "insightface", "error": None},
                    "ocr_extractor": {"loaded": True, "name": "paddleocr", "error": None}
                },
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }


# ============================================================================
# Liveness Detection Schemas
# ============================================================================

class ChallengeType(str, Enum):
    """Types of liveness challenges."""
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class ChallengeStatus(str, Enum):
    """Challenge validation status."""
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"
    EXPIRED = "expired"
    INVALID = "invalid"


class ChallengeResponse(BaseModel):
    """Liveness challenge generation response (supports single or multi-challenge)."""
    challenge_id: str = Field(description="Unique challenge identifier")
    multi_challenge: bool = Field(default=False, description="Whether this is a multi-challenge session")
    
    # Single challenge fields (for backward compatibility)
    challenge_type: Optional[ChallengeType] = Field(default=None, description="Type of challenge (single mode)")
    question: Optional[str] = Field(default=None, description="Challenge question text (single mode)")
    instruction: Optional[str] = Field(default=None, description="User-friendly instruction (single mode)")
    
    # Multi-challenge fields
    challenge_types: Optional[List[ChallengeType]] = Field(default=None, description="List of challenge types (multi mode)")
    questions: Optional[List[str]] = Field(default=None, description="List of challenge questions (multi mode)")
    instructions: Optional[List[str]] = Field(default=None, description="List of user-friendly instructions (multi mode)")
    
    timestamp: float = Field(description="Challenge creation timestamp")
    expires_at: float = Field(description="Challenge expiration timestamp")
    nonce: str = Field(description="Nonce for replay protection")
    signature: str = Field(description="HMAC signature for challenge integrity")

    class Config:
        json_schema_extra = {
            "example": {
                "challenge_id": "550e8400-e29b-41d4-a716-446655440000",
                "multi_challenge": True,
                "challenge_types": ["blink", "turn_left"],
                "questions": ["blink eyes", "turn face left"],
                "instructions": ["Blink your eyes once", "Turn your face to the left"],
                "timestamp": 1697022600.0,
                "expires_at": 1697022630.0,
                "nonce": "a1b2c3d4e5f6",
                "signature": "abc123..."
            }
        }


class LivenessDetectionResult(BaseModel):
    """Individual frame detection result."""
    blinks: int = Field(description="Number of blinks detected")
    orientation: Optional[str] = Field(description="Face orientation: left, right, or None")
    face_detected: bool = Field(description="Whether face was detected")
    ear_value: float = Field(description="Eye Aspect Ratio value")
    is_blinking: bool = Field(description="Whether eyes are currently blinking")


class LivenessVerificationRequest(BaseModel):
    """Request to verify liveness challenge."""
    challenge_id: str = Field(description="Challenge ID to verify")
    frames: list = Field(description="Base64-encoded image frames (list of strings)")

    class Config:
        json_schema_extra = {
            "example": {
                "challenge_id": "550e8400-e29b-41d4-a716-446655440000",
                "frames": ["data:image/jpeg;base64,/9j/4AAQ...", "data:image/jpeg;base64,/9j/4AAQ..."]
            }
        }


class LivenessVerificationResponse(BaseModel):
    """Liveness challenge verification response."""
    challenge_id: str = Field(description="Challenge ID that was verified")
    status: ChallengeStatus = Field(description="Verification status")
    message: str = Field(description="Human-readable result message")
    detection_results: Dict[str, Any] = Field(description="Detailed detection results")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "challenge_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pass",
                "message": "All challenges completed: blink, turn left",
                "detection_results": {
                    "blinks": 2,
                    "orientation": "left",
                    "orientations": ["left", "left", None],
                    "face_detected": True
                },
                "processing_time_ms": 1250,
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }


class LivenessBatchRequest(BaseModel):
    """Request for batch liveness detection (without challenge)."""
    frames: list = Field(description="Base64-encoded image frames (list of strings)")
    initial_blink_count: int = Field(default=0, description="Initial blink count for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "frames": ["data:image/jpeg;base64,/9j/4AAQ...", "data:image/jpeg;base64,/9j/4AAQ..."],
                "initial_blink_count": 0
            }
        }


class LivenessBatchResponse(BaseModel):
    """Batch liveness detection response."""
    total_blinks: int = Field(description="Total new blinks detected in batch")
    final_blink_count: int = Field(description="Final cumulative blink count")
    orientations: list = Field(description="Detected orientations per frame")
    face_detection_ratio: float = Field(description="Ratio of frames with face detected")
    results: list = Field(description="Per-frame detection results")
    frame_count: int = Field(description="Number of frames processed")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "total_blinks": 2,
                "final_blink_count": 2,
                "orientations": [None, None, "left"],
                "face_detection_ratio": 0.85,
                "results": [],
                "frame_count": 30,
                "processing_time_ms": 1200,
                "timestamp": "2024-10-11T10:30:00Z"
            }
        }