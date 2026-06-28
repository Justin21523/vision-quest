from typing import Optional

from pydantic import BaseModel, Field


class CaptionRequest(BaseModel):
    max_length: int = Field(50, ge=10, le=200)
    num_beams: int = Field(3, ge=1, le=10)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    enable_safety: bool = True


class CaptionResponse(BaseModel):
    caption: str
    confidence: float = Field(ge=0.0, le=1.0)
    safety_score: Optional[float] = None
    is_safe: bool = True
    model: str
    model_used: str
    processing_time_ms: float = 0.0


class BatchCaptionRequest(BaseModel):
    max_length: int = Field(50, ge=10, le=200)
    num_beams: int = Field(3, ge=1, le=10)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    enable_safety: bool = True


class BatchCaptionResponse(BaseModel):
    results: list[CaptionResponse]
    total: int
    successful: int
    failed: int
