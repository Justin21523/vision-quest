# backend/app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="訊息角色")
    content: str = Field(..., description="訊息內容")
    timestamp: Optional[str] = Field(None, description="時間戳")


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    model: Optional[str] = Field("qwen", description="使用的模型")
    max_tokens: int = Field(default=150, ge=1, le=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tools: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = Field(False, description="是否串流回應")


class ChatResponse(BaseModel):
    message: ChatMessage
    tokens_used: int
    finish_reason: str
    model_used: str
    processing_time_ms: float = Field(default=0.0)
    model: str = Field(..., description="使用的模型")
    usage: Optional[dict]
