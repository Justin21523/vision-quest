# backend/app/schemas/vqa.py
from pydantic import BaseModel, Field
from typing import Optional
import regex as re


class VQARequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    language: str = Field(default="en", regex="^(en|zh-tw|zh-cn)$")
    max_length: int = Field(default=100, ge=10, le=300)


class VQAResponse(BaseModel):
    answer: str
    raw_answer: Optional[str] = None
    question: str
    language: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_used: str
