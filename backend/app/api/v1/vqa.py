# backend/app/api/v1/vqa.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Form

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.services.vqa_service import VQAService
from backend.app.schemas.vqa import VQARequest, VQAResponse
from backend.app.core.security import validate_image
import logging

router = APIRouter(prefix="/vqa", tags=["Vision - VQA"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=VQAResponse)
async def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(...),
    lang: str = Form(default="en"),
    max_length: int = Form(default=100),
):
    """
    Visual Question Answering using LLaVA or Qwen-VL
    Supports Chinese (zh-tw/zh-cn) and English
    """
    try:
        # Validate inputs
        validated_image = await validate_image(file)
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Initialize service
        vqa_service = VQAService()

        # Generate answer
        result = await vqa_service.answer_question(
            image=validated_image,
            question=question,
            language=lang,
            max_length=max_length,
        )

        logger.info(f"VQA completed for {file.filename}, question: {question[:50]}...")
        return VQAResponse(**result)

    except Exception as e:
        logger.error(f"VQA failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
