"""Pydantic schemas for the KYC domain."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    ERROR = "error"


class OCRFields(BaseModel):
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
    document_type: str = Field(description="Detected document type")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall OCR confidence")
    extracted_text: str = Field(description="Full raw text extracted")
    fields: OCRFields = Field(description="Structured extracted fields")


class SimilarityMetrics(BaseModel):
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    euclidean_distance: float = Field(ge=0.0)


class FaceMatchData(BaseModel):
    verified: bool = Field(description="Whether faces match")
    confidence: float = Field(ge=0.0, le=1.0, description="Match confidence score")
    similarity_metrics: SimilarityMetrics
    threshold_used: float
    message: str


class KYCVerificationResponse(BaseModel):
    verification_status: VerificationStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    face_match_score: float = Field(ge=0.0, le=1.0)
    ocr_data: OCRData
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    face_verification_details: Optional[FaceMatchData] = None
    error_message: Optional[str] = None


class OCROnlyResponse(BaseModel):
    ocr_data: OCRData
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelStatus(BaseModel):
    loaded: bool
    name: str
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    status: str = Field(description="healthy | degraded | unhealthy")
    version: str
    models: Dict[str, ModelStatus]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChallengeType(str, Enum):
    BLINK = "blink"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class ChallengeStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"
    EXPIRED = "expired"
    INVALID = "invalid"


class ChallengeResponse(BaseModel):
    challenge_id: str
    multi_challenge: bool = False
    challenge_type: Optional[ChallengeType] = None
    question: Optional[str] = None
    instruction: Optional[str] = None
    challenge_types: Optional[List[ChallengeType]] = None
    questions: Optional[List[str]] = None
    instructions: Optional[List[str]] = None
    timestamp: float
    expires_at: float
    nonce: str
    signature: Optional[str] = None


class LivenessVerificationRequest(BaseModel):
    challenge_id: str
    frames: List[str] = Field(description="Base64-encoded image frames")


class LivenessVerificationResponse(BaseModel):
    challenge_id: str
    status: ChallengeStatus
    message: str
    detection_results: Dict[str, Any]
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LivenessBatchRequest(BaseModel):
    frames: List[str]
    initial_blink_count: int = 0


class LivenessBatchResponse(BaseModel):
    total_blinks: int
    final_blink_count: int
    orientations: list
    face_detection_ratio: float
    results: list
    frame_count: int
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
