# backend/app/schemas/caption.py
from pydantic import BaseModel, Field
from typing import Optional


class CaptionRequest(BaseModel):
    max_length: int = Field(default=50, ge=10, le=200)
    num_beams: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)


class CaptionResponse(BaseModel):
    caption: str
    raw_caption: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float = Field(default=0.0)
    model_used: str
