# backend/app/api/v1/caption.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.services.caption_service import CaptionService
from backend.app.schemas.caption import CaptionRequest, CaptionResponse
from backend.app.core.security import validate_image
import logging

router = APIRouter(prefix="/caption", tags=["Vision - Caption"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=CaptionResponse)
async def generate_caption(
    file: UploadFile = File(...),
    max_length: int = 50,
    num_beams: int = 5,
    temperature: float = 1.0,
):
    """
    Generate image caption using BLIP-2
    """
    try:
        # Validate image
        validated_image = await validate_image(file)

        # Initialize service
        caption_service = CaptionService()

        # Generate caption
        result = await caption_service.generate_caption(
            image=validated_image,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
        )

        logger.info(f"Caption generated successfully for {file.filename}")
        return CaptionResponse(**result)

    except Exception as e:
        logger.error(f"Caption generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
