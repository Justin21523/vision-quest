from typing import Any, Dict

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from app.services.agent_service import AgentService

router = APIRouter()
agent_service = AgentService()


class ToolRequest(BaseModel):
    tool: str
    data: Dict[str, Any]


@router.post("/execute")
async def execute_agent_tool(request: ToolRequest):
    return await agent_service.execute_task(request.tool, request.data)


@router.post("/chat")
async def agent_chat(prompt: str = Form(...)):
    return await agent_service.run_reasoning_loop(prompt)


@router.post("/process-image")
async def process_image_auto(file: UploadFile = File(...), remove_bg: bool = Form(True)):
    contents = await file.read()
    result: Dict[str, Any] = {}
    if remove_bg:
        result["background_removal"] = await agent_service.execute_task(
            "background_removal",
            {"image_bytes": contents, "filename": file.filename or "asset.png"},
        )
    return {"status": "success", "original_filename": file.filename, "results": result}
