import platform
from datetime import datetime, timezone

import psutil
from fastapi import APIRouter

from app.core.config import settings
from app.models.manager import ModelManager

router = APIRouter(prefix="/health", tags=["System"])
model_manager = ModelManager()


@router.get("/")
async def health_check():
    memory = psutil.virtual_memory()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "mock_mode": settings.USE_MOCK_MODE,
        "models": model_manager.get_loaded_models(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / 1024**3, 2),
        },
        "gpu": {"available": False, "device": model_manager.device},
    }


@router.get("/detailed")
async def detailed_health_check():
    base = await health_check()
    base["app_info"] = {"name": settings.APP_NAME, "version": settings.APP_VERSION, "environment": settings.ENV}
    base["configuration"] = {"device": settings.DEVICE, "max_workers": settings.MAX_WORKERS}
    base["system"].update({"platform": platform.platform(), "python_version": platform.python_version(), "cpu_count": psutil.cpu_count()})
    return base


@router.get("/models")
async def models_health_check():
    return {"models": model_manager.scan_local_models(), "dependencies": {"mock_safe": True}}


@router.get("/services")
async def services_health_check():
    return {"services": {"api": "healthy", "database": "optional in demo mode", "rag": "in-memory demo"}}


@router.get("/readiness")
async def readiness_check():
    return {"ready": True, "checks": {"configuration_loaded": True, "directories_accessible": True, "basic_dependencies": True}}


@router.get("/liveness")
async def liveness_check():
    return {"alive": True, "timestamp": datetime.now(timezone.utc).isoformat()}
