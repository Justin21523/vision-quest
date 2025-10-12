# backend/app/services/caption_service.py
import torch
from PIL import Image
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.models.manager import ModelManager
from backend.app.utils.image_processing import preprocess_image
from backend.app.utils.safety import content_filter
from app.models.vlm_models import get_caption_pipeline
from app.utils.image_processing import preprocess_image, validate_image
from app.utils.safety import NSFWFilter


logger = logging.getLogger(__name__)


class CaptionService:
    def __init__(self):
        self.model_manager = ModelManager()
        self.processor = None
        self.model = None
        self.nsfw_filter = NSFWFilter()

    async def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 50,
        num_beams: int = 3,
        temperature: float = 1.0,
        enable_safety: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate caption for image with optional safety check

        Returns:
            {
                "caption": str,
                "confidence": float,
                "safety_score": Optional[float],
                "is_safe": bool,
            }
        """
        # Validate image
        validate_image(image)

        # Safety check if enabled
        is_safe = True
        safety_score = None
        if enable_safety:
            safety_result = await self.nsfw_filter.check_image(image)
            is_safe = safety_result["is_safe"]
            safety_score = safety_result["score"]

            if not is_safe:
                return {
                    "caption": "[Content filtered - inappropriate image]",
                    "confidence": 0.0,
                    "safety_score": safety_score,
                    "is_safe": False,
                }

        # Preprocess image
        processed_image = preprocess_image(
            image,
            target_size=(384, 384),  # BLIP-2 default
            normalize=False,
        )

        # Generate caption
        pipeline = get_caption_pipeline()
        caption = pipeline.generate_caption(
            processed_image,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
        )

        return {
            "caption": caption,
            "confidence": 1.0,  # TODO: Add actual confidence from model
            "safety_score": safety_score,
            "is_safe": is_safe,
        }

    async def batch_caption(
        self,
        images: list[Image.Image],
        **kwargs,
    ) -> list[Dict[str, Any]]:
        """Generate captions for multiple images"""
        results = []
        for image in images:
            result = await self.generate_caption(image, **kwargs)
            results.append(result)
        return results
