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
