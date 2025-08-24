# backend/app/schemas/common.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=1.0)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class BatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    batch_size: int = Field(default=5, ge=1, le=20)
    priority: int = Field(default=1, ge=1, le=10)


class HealthResponse(BaseModel):
    status: str
    models: Dict[str, str]
    system: Dict[str, float]
    gpu: Dict[str, Any]
    version: str
