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
            {"id": "1", "data": {"label": "Caption: extract image metadata"}, "position": {"x": 0, "y": 0}},
            {"id": "2", "data": {"label": "VQA: answer visual question"}, "position": {"x": 0, "y": 0}},
            {"id": "3", "data": {"label": "RAG: retrieve project context"}, "position": {"x": 0, "y": 0}},
            {"id": "4", "data": {"label": "Game: update branch state"}, "position": {"x": 0, "y": 0}},
        ],
        "edges": [
            {"id": "e1-2", "source": "1", "target": "2"},
            {"id": "e2-3", "source": "2", "target": "3"},
            {"id": "e3-4", "source": "3", "target": "4"},
        ],
    }
