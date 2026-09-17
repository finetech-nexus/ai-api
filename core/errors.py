"""Shared HTTP error models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "No face detected in ID document",
                "details": {"confidence_threshold": 0.6},
                "timestamp": "2024-10-11T10:30:00Z",
            }
        }
    }
