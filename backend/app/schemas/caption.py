# backend/app/schemas/caption.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class CaptionRequest(BaseModel):
    """Request model for caption generation"""

    max_length: int = Field(50, ge=10, le=200, description="Maximum caption length")
    num_beams: int = Field(3, ge=1, le=10, description="Beam search width")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    enable_safety: bool = Field(True, description="Enable NSFW content filter")


class CaptionResponse(BaseModel):
    """Response model for caption generation"""

    caption: str = Field(..., description="Generated caption text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    safety_score: Optional[float] = Field(None, description="NSFW safety score (0-1)")
    is_safe: bool = Field(True, description="Whether content passed safety check")
    model: str = Field(..., description="Model identifier used for generation")

    class Config:
        json_schema_extra = {
            "example": {
                "caption": "A beautiful sunset over the ocean with orange and pink clouds",
                "confidence": 0.95,
                "safety_score": 0.98,
                "is_safe": True,
                "model": "blip2-opt-2.7b",
            }
        }


class BatchCaptionRequest(BaseModel):
    """Request model for batch caption generation"""

    max_length: int = Field(50, ge=10, le=200)
    num_beams: int = Field(3, ge=1, le=10)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    enable_safety: bool = Field(True)


class BatchCaptionResponse(BaseModel):
    """Response model for batch caption generation"""

    results: list[CaptionResponse]
    total: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successful captions")
    failed: int = Field(..., description="Number of failed captions")
