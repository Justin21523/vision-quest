from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "VisionQuest"
    APP_VERSION: str = "0.2.0-demo"
    ENV: str = "development"
    DEBUG: bool = True
    USE_MOCK_MODE: bool = True

    API_PREFIX: str = "/api/v1"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    FRONTEND_PORT: int = 3000

    DEVICE: str = "auto"
    MAX_WORKERS: int = 4
    MAX_BATCH_SIZE: int = 10

    DEFAULT_CAPTION_MODEL: str = "demo-blip2"
    DEFAULT_VQA_MODEL: str = "demo-vqa"
    DEFAULT_LLM_MODEL: str = "demo-qwen"

    UPLOAD_DIR: str = "./data/uploads"
    OUTPUT_DIR: str = "./data/outputs"
    KB_DIR: str = "./data/kb"
    MODEL_CACHE_DIR: str = "./models"

    ENABLE_NSFW_FILTER: bool = True
    ENABLE_FACE_BLUR: bool = False
    CONTENT_FILTER_LEVEL: str = "medium"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/visionquest"
    LOG_LEVEL: str = "INFO"

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        return [
            f"http://localhost:{self.FRONTEND_PORT}",
            f"http://127.0.0.1:{self.FRONTEND_PORT}",
            f"http://localhost:{self.API_PORT}",
            f"http://127.0.0.1:{self.API_PORT}",
            "https://justin21523.github.io",
        ]

    @field_validator("ENV")
    @classmethod
    def validate_env(cls, value: str) -> str:
        if value not in {"development", "testing", "production"}:
            raise ValueError("ENV must be development, testing, or production")
        return value

    @field_validator("DEVICE")
    @classmethod
    def validate_device(cls, value: str) -> str:
        valid = {"auto", "cpu", "cuda", "mps"}
        if value not in valid and not value.startswith("cuda:"):
            raise ValueError("DEVICE must be auto, cpu, cuda, mps, or cuda:N")
        return value

    @field_validator("CONTENT_FILTER_LEVEL")
    @classmethod
    def validate_filter_level(cls, value: str) -> str:
        if value not in {"off", "low", "medium", "high"}:
            raise ValueError("CONTENT_FILTER_LEVEL must be off, low, medium, or high")
        return value

    @field_validator("UPLOAD_DIR", "OUTPUT_DIR", "KB_DIR", "MODEL_CACHE_DIR")
    @classmethod
    def ensure_directories(cls, value: str) -> str:
        Path(value).mkdir(parents=True, exist_ok=True)
        return value

    @property
    def is_development(self) -> bool:
        return self.ENV == "development"

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
