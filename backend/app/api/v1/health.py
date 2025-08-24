# backend/app/api/v1/health.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.models.manager import ModelManager
import psutil
import torch

router = APIRouter(prefix="/health", tags=["System"])


@router.get("/")
async def health_check():
    """
    System health check with GPU/CPU status
    """
    try:
        # Model status
        model_manager = ModelManager()
        models_loaded = model_manager.get_loaded_models()

        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            }
        else:
            gpu_info = {"available": False}

        return JSONResponse(
            {
                "status": "healthy",
                "models": models_loaded,
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / 1024**3,
                },
                "gpu": gpu_info,
                "version": "0.1.0",
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )
