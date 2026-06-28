from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings


MODEL_PATHS = {
    "LLM": "/mnt/c/ai_models/language/llm/",
    "VLM": "/mnt/c/ai_models/language/vlm/",
    "DIFFUSION": "/mnt/c/ai_models/diffusion/",
}


class ModelManager:
    def __init__(self) -> None:
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.mock_mode = settings.USE_MOCK_MODE
        self.device = "cpu (mock-safe demo)" if self.mock_mode else self._detect_device()

    def _detect_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def scan_local_models(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.mock_mode:
            return {
                "LLM": [{"name": "demo-qwen", "path": "mock://qwen", "size_gb": 0.0}],
                "VLM": [{"name": "demo-vqa", "path": "mock://vqa", "size_gb": 0.0}],
                "DIFFUSION": [{"name": "demo-asset-lab", "path": "mock://diffusion", "size_gb": 0.0}],
            }

        indicators = {".safetensors", ".bin", ".gguf", "config.json", "model_index.json"}
        results: Dict[str, List[Dict[str, Any]]] = {"LLM": [], "VLM": [], "DIFFUSION": []}
        for category, root_path in MODEL_PATHS.items():
            root = Path(root_path)
            if not root.exists():
                continue
            for entry in root.iterdir():
                files = entry.iterdir() if entry.is_dir() else [entry]
                if any(f.name in indicators or f.suffix in indicators for f in files):
                    results[category].append(
                        {"name": entry.name, "path": str(entry), "size_gb": round(self._size(entry) / 1024**3, 2)}
                    )
        return results

    def _size(self, path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.glob("**/*") if item.is_file())

    def get_loaded_models(self) -> Dict[str, str]:
        if not self.loaded_models and self.mock_mode:
            return {
                "demo-qwen": f"ready on {self.device}",
                "demo-vqa": f"ready on {self.device}",
                "demo-rag": "in-memory vector demo",
            }
        return {name: f"loaded on {self.device}" for name in self.loaded_models}

    def load_model(self, model_name: str, model_path: str, category: str) -> bool:
        self.loaded_models[model_name] = {"path": model_path, "category": category}
        return True

    def unload_model(self, model_name: str) -> bool:
        return self.loaded_models.pop(model_name, None) is not None

    def get_memory_usage(self) -> Dict[str, float]:
        return {"gpu_allocated_gb": 0.0, "gpu_reserved_gb": 0.0} if self.mock_mode else {}
