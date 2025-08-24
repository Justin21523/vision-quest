# backend/app/models/manager.py
import torch
from typing import Dict, Optional, Any
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Unified model manager for VLM and LLM models
    Handles memory optimization and device placement
    """

    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.device = self._get_optimal_device()
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _get_optimal_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_loaded_models(self) -> Dict[str, str]:
        """Return status of loaded models"""
        return {name: f"loaded on {self.device}" for name in self.loaded_models.keys()}

    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}

        if torch.cuda.is_available():
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            memory_info["gpu_max_memory_gb"] = (
                torch.cuda.max_memory_allocated() / 1024**3
            )

        return memory_info
