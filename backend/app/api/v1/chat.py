from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["NLP - Chat"])
chat_service = ChatService()


@router.post("/", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in chat_service.generate_stream(request)),
                media_type="text/event-stream",
            )
        return await chat_service.generate_response(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}")


@router.get("/models")
async def get_available_models():
    return {
        "models": [
            {"id": "demo-qwen", "name": "Demo Qwen-compatible responder"},
            {"id": "demo-llama", "name": "Demo Llama-compatible responder"},
        ]
    }
