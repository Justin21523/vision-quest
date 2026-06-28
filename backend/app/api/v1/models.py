from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any
from app.models.manager import ModelManager
from pydantic import BaseModel

router = APIRouter()
model_manager = ModelManager()

class LoadModelRequest(BaseModel):
    name: str
    path: str
    category: str

@router.get("/scan")
async def scan_models():
    """Scan local directories for models"""
    return model_manager.scan_local_models()

@router.get("/loaded")
async def get_loaded():
    """Get currently loaded models"""
    return model_manager.get_loaded_models()

@router.post("/load")
async def load_model(request: LoadModelRequest):
    """Load a specific model"""
    success = model_manager.load_model(request.name, request.path, request.category)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model {request.name}")
    return {"status": "success", "message": f"Model {request.name} loaded"}

@router.post("/unload/{name}")
async def unload_model(name: str):
    """Unload a specific model"""
    success = model_manager.unload_model(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {name} not found or not loaded")
    return {"status": "success", "message": f"Model {name} unloaded"}

@router.get("/memory")
async def get_memory():
    """Get GPU memory usage"""
    return model_manager.get_memory_usage()
