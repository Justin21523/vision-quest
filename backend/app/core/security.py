# backend/app/core/security.py
from fastapi import UploadFile, HTTPException
from PIL import Image
import io
from typing import List
import logging

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


async def validate_image(file: UploadFile) -> Image.Image:
    """
    Validate and process uploaded image file
    """
    try:
        # Check file type
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type. Allowed: {ALLOWED_IMAGE_TYPES}",
            )

        # Check file size
        contents = await file.read()
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large. Max size: {MAX_IMAGE_SIZE // (1024*1024)}MB",
            )

        # Try to load image
        image = Image.open(io.BytesIO(contents))

        # Basic validation
        if image.size[0] * image.size[1] > 4096 * 4096:  # Max 16MP
            raise HTTPException(
                status_code=400, detail="Image resolution too high. Max: 4096x4096"
            )

        logger.info(f"Image validated: {file.filename}, size: {image.size}")
        return image

    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail="Invalid image file")
