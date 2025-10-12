# backend/app/api/v1/caption.py
"""
Caption API Endpoints
POST /api/v1/caption - Generate caption for single image
"""
from fastapi import APIRouter, File, Query, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from PIL import Image
import io
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.services.caption_service import CaptionService
from backend.app.schemas.caption import CaptionRequest, CaptionResponse
from backend.app.core.security import validate_image
from app.utils.image_processing import ImageProcessingError


router = APIRouter(prefix="/caption", tags=["caption"])
logger = logging.getLogger(__name__)

caption_service = CaptionService()


@router.post(
    "",
    response_model=CaptionResponse,
    summary="Generate image caption",
    description="Generate a descriptive caption for an uploaded image using BLIP-2",
)
async def generate_caption(
    file: UploadFile = File(..., description="Image file (JPEG/PNG/WEBP)"),
    max_length: int = Query(50, ge=10, le=200, description="Maximum caption length"),
    num_beams: int = Query(3, ge=1, le=10, description="Beam search width"),
    temperature: float = Query(1.0, ge=0.1, le=2.0, description="Sampling temperature"),
    enable_safety: bool = Query(True, description="Enable NSFW filter"),
) -> CaptionResponse:
    """
    Generate caption for uploaded image

    Args:
        file: Image file upload
        max_length: Maximum caption length in tokens
        num_beams: Beam search parameter
        temperature: Sampling temperature
        enable_safety: Enable content safety filter

    Returns:
        CaptionResponse with generated caption and metadata
    """
    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Generate caption
        result = await caption_service.generate_caption(
            image=image,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            enable_safety=enable_safety,
        )

        return CaptionResponse(
            caption=result["caption"],
            confidence=result["confidence"],
            safety_score=result.get("safety_score"),
            is_safe=result["is_safe"],
            model="blip2-opt-2.7b",
        )

    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Caption generation failed: {str(e)}",
        )


@router.get(
    "/health",
    summary="Check caption service health",
)
async def caption_health():
    """Check if caption model is loaded and ready"""
    try:
        from app.models.vlm_models import _caption_pipeline

        is_ready = _caption_pipeline is not None
        return JSONResponse(
            {
                "status": "ready" if is_ready else "not_loaded",
                "model": "blip2-opt-2.7b" if is_ready else None,
            }
        )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "detail": str(e)},
            status_code=500,
        )
