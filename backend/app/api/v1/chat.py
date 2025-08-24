# backend/app/api/v1/chat.py
from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.services.chat_service import ChatService
from backend.app.schemas.chat import ChatRequest, ChatResponse
import logging

router = APIRouter(prefix="/chat", tags=["NLP - Chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """對話完成端點"""
    try:
        if request.stream:

            def generate():
                for chunk in chat_service.generate_stream(request):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            response = chat_service.generate_response(request)
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get("/chat/models")
async def get_available_models():
    """取得可用模型列表"""
    return {
        "models": [
            {"id": "qwen", "name": "Qwen2-7B-Instruct"},
            {"id": "llama", "name": "Llama-3.1-8B-Instruct"},
        ]
    }


@router.post("/", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Text generation and chat using Qwen/Llama models
    """
    try:
        # Initialize service
        chat_service = ChatService()

        # Generate response
        result = await chat_service.generate_response(
            messages=request.messages,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            tools=request.tools,
        )

        logger.info(f"Chat response generated, tokens: {result.get('tokens_used', 0)}")
        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Chat generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
