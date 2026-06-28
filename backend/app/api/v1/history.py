from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SaveSlotCreate(BaseModel):
    slot_name: str
    metadata: Dict[str, Any]
    rag_context_ids: List[int]
    game_state: Dict[str, Any]


@router.post("/saves")
async def create_save(request: SaveSlotCreate):
    return {"status": "success", "save_id": 1, "slot_name": request.slot_name, "demo_mode": True}


@router.get("/dag")
async def get_history_dag():
    return {
        "nodes": [
            {"id": "upload", "data": {"label": "Upload image", "kind": "input"}, "position": {"x": 0, "y": 80}},
            {"id": "caption", "data": {"label": "Caption + objects", "kind": "vision"}, "position": {"x": 180, "y": 20}},
            {"id": "vqa", "data": {"label": "VQA reasoning", "kind": "vision"}, "position": {"x": 180, "y": 140}},
            {"id": "rag", "data": {"label": "RAG evidence", "kind": "knowledge"}, "position": {"x": 380, "y": 80}},
            {"id": "agent", "data": {"label": "Agent tool trace", "kind": "agent"}, "position": {"x": 580, "y": 80}},
            {"id": "game", "data": {"label": "Adventure state", "kind": "state"}, "position": {"x": 780, "y": 80}},
            {"id": "save", "data": {"label": "Save branch", "kind": "history"}, "position": {"x": 980, "y": 80}},
        ],
        "edges": [
            {"id": "upload-caption", "source": "upload", "target": "caption"},
            {"id": "upload-vqa", "source": "upload", "target": "vqa"},
            {"id": "caption-rag", "source": "caption", "target": "rag"},
            {"id": "vqa-rag", "source": "vqa", "target": "rag"},
            {"id": "rag-agent", "source": "rag", "target": "agent"},
            {"id": "agent-game", "source": "agent", "target": "game"},
            {"id": "game-save", "source": "game", "target": "save"},
        ],
        "summary": "Mock-safe DAG showing how visual evidence becomes cited context, agent actions, and a saved branch.",
    }
