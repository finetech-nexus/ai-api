"""Pydantic schemas for the AML domain (future)."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class AmlScreenStatus(str, Enum):
    CLEAR = "clear"
    POTENTIAL_MATCH = "potential_match"
    CONFIRMED_MATCH = "confirmed_match"
    ERROR = "error"


class AmlParty(BaseModel):
    full_name: str = Field(description="Name to screen")
    date_of_birth: Optional[str] = None
    nationality: Optional[str] = None
    document_number: Optional[str] = None


class AmlScreenRequest(BaseModel):
    party: AmlParty
    lists: List[str] = Field(
        default_factory=lambda: ["sanctions", "pep", "adverse_media"],
        description="Watchlists to query",
    )


class AmlScreenResponse(BaseModel):
    status: AmlScreenStatus
    message: str
    matches: List[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AmlHealthResponse(BaseModel):
    domain: str = "aml"
    status: str = "not_implemented"
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
