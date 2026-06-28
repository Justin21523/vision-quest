from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from app.services.game_service import GameService
from pydantic import BaseModel

router = APIRouter()
game_service = GameService()

class NewGameRequest(BaseModel):
    scenario: str
    persona: Dict[str, Any]

class ActionRequest(BaseModel):
    action: str

@router.post("/start")
async def start_game(request: NewGameRequest):
    """開始新遊戲"""
    return await game_service.start_new_game(request.scenario, request.persona)

@router.post("/act")
async def game_act(request: ActionRequest):
    """玩家動作"""
    return await game_service.take_action(request.action)

@router.get("/state")
async def get_state():
    """獲取當前遊戲狀態"""
    return game_service.game_state
