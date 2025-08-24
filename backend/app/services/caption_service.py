# backend/app/services/caption_service.py
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.models.manager import ModelManager
from backend.app.utils.image_processing import preprocess_image
from backend.app.utils.safety import content_filter
import asyncio
import logging

logger = logging.getLogger(__name__)


class CaptionService:
    def __init__(self):
        self.model_manager = ModelManager()
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load BLIP-2 model with fp16 optimization"""
        try:
            model_name = "Salesforce/blip2-opt-2.7b"

            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

            # Enable memory efficient attention if available
            if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                self.model.enable_xformers_memory_efficient_attention()

            logger.info(f"BLIP-2 model loaded successfully on {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            raise

    async def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 50,
        num_beams: int = 5,
        temperature: float = 1.0,
    ) -> dict:
        """
        Generate image caption using BLIP-2
        """
        try:
            # Preprocess image
            processed_image = preprocess_image(image, target_size=(384, 384))

            # Prepare inputs
            inputs = self.processor(processed_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=True if temperature > 1.0 else False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

            # Apply content filter
            filtered_caption = content_filter(caption)

            return {
                "caption": filtered_caption,
                "raw_caption": caption,
                "confidence": 0.85,  # Placeholder
                "processing_time_ms": 0,  # Add timing if needed
                "model_used": "blip2-opt-2.7b",
            }

        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            raise
