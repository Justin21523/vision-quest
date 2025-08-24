# backend/app/core/config.py
"""
Core configuration management for VisionQuest application.
Handles environment variables, settings validation, and application configuration.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # === Application Info ===
    APP_NAME: str = Field("VisionQuest", env="APP_NAME")
    APP_VERSION: str = Field("0.1.0", env="APP_VERSION")
    ENV: str = Field("development", env="ENV")
    DEBUG: bool = Field(True, env="DEBUG")

    # === API Configuration ===
    API_PREFIX: str = Field("/api/v1", env="API_PREFIX")
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React dev
        "http://localhost:7860",  # Gradio
        "http://localhost:8000",  # FastAPI
    ]

    # === Hardware Configuration ===
    DEVICE: str = Field("auto", env="DEVICE")
    MAX_WORKERS: int = Field(4, env="MAX_WORKERS")
    MAX_BATCH_SIZE: int = Field(10, env="MAX_BATCH_SIZE")
    MODEL_CACHE_DIR: str = "./models"

    # === Model Configuration ===
    DEFAULT_CAPTION_MODEL: str = Field("blip2", env="DEFAULT_CAPTION_MODEL")
    DEFAULT_VQA_MODEL: str = Field("llava", env="DEFAULT_VQA_MODEL")
    DEFAULT_LLM_MODEL: str = Field("qwen", env="DEFAULT_LLM_MODEL")

    # === Storage Paths ===
    UPLOAD_DIR: str = Field("./data/uploads", env="UPLOAD_DIR")
    OUTPUT_DIR: str = Field("./data/outputs", env="OUTPUT_DIR")
    KB_DIR: str = Field("./data/kb", env="KB_DIR")
    MODEL_CACHE_DIR: str = Field("./models", env="MODEL_CACHE_DIR")

    # === Security Settings ===
    ENABLE_NSFW_FILTER: bool = Field(True, env="ENABLE_NSFW_FILTER")
    ENABLE_FACE_BLUR: bool = Field(False, env="ENABLE_FACE_BLUR")
    ENABLE_FACE_BLUR: bool = False
    CONTENT_FILTER_LEVEL: str = Field("medium", env="CONTENT_FILTER_LEVEL")
    MAX_FILE_SIZE: int = Field(10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB

    # === Database ===
    DATABASE_URL: str = Field("sqlite:///./data/visionquest.db", env="DATABASE_URL")

    # Performance Settings
    USE_FP16: bool = True
    USE_ATTENTION_SLICING: bool = True
    CPU_OFFLOAD: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"

    @field_validator("ENV", mode="after")
    def validate_env(cls, v):
        if v not in ["development", "testing", "production"]:
            raise ValueError("ENV must be development, testing, or production")
        return v

    @field_validator("DEVICE", mode="after")
    def validate_device(cls, v):
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if v not in valid_devices and not v.startswith("cuda:"):
            raise ValueError(f"DEVICE must be one of {valid_devices} or cuda:N format")
        return v

    @field_validator("CONTENT_FILTER_LEVEL", mode="after")
    def validate_filter_level(cls, v):
        if v not in ["off", "low", "medium", "high"]:
            raise ValueError("CONTENT_FILTER_LEVEL must be off, low, medium, or high")
        return v

    @field_validator(
        "UPLOAD_DIR", "OUTPUT_DIR", "KB_DIR", "MODEL_CACHE_DIR", mode="after"
    )
    def ensure_directories(cls, v):
        """Ensure directories exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @property
    def is_development(self) -> bool:
        return self.ENV == "development"

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
