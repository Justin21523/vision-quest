# backend/app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatMessage(BaseModel):
    role: str = Field(regex="^(system|user|assistant)$")
    content: str = Field(min_length=1)
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    max_tokens: int = Field(default=150, ge=1, le=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tools: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    message: ChatMessage
    tokens_used: int
    finish_reason: str
    model_used: str
    processing_time_ms: float = Field(default=0.0)
