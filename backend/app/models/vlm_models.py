"""
Vision-Language Model Pipelines
Supports: BLIP-2 (caption), LLaVA/Qwen-VL (VQA)
"""

import torch
import logging
from typing import Optional, Dict, Any
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from app.core.shared_cache import get_shared_cache
from app.core.config import get_config


logger = logging.getLogger(__name__)


class CaptionPipeline:
    """BLIP-2 image captioning pipeline"""

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        quantization: str = "none",
        **kwargs,
    ):
        self.cache = get_shared_cache()
        self.config = get_config()
        self.model_name = model_name
        self.device = self.config.device.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = (
            self.config.device.torch_dtype if self.device == "cuda" else torch.float32
        )

        cache_dir = self.cache.get_path("MODELS_VLM")

        kwargs_model: Dict[str, Any] = dict(
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # Optional quantization
        if quantization == "4bit":
            kwargs_model["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            kwargs_model["load_in_8bit"] = True

        # --- generation defaults (align to HuggingFaceClient.gen_kwargs) ---
        self.generation_defaults: Dict[str, Any] = dict(
            max_new_tokens=60,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            num_beams=1,
            repetition_penalty=1.05,
            length_penalty=1.0,
            use_cache=True,
            pad_token_id=None,  # will be filled after tokenizer is ready
            eos_token_id=None,
        )
        self.kwargs_model = kwargs_model

        logger.info(f"Loading BLIP-2 caption model from {model_name}...")
        # load processor
        self.processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)

        # load model
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, **kwargs_model
        )

        if hasattr(self.model, "enable_attention_slicing") and self.device == "cuda":
            self.model.enable_attention_slicing()  # type: ignore

        self.model.eval()
        logger.info(f"✅ BLIP-2 loaded on {self.device}")

    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 50,
        num_beams: int = 3,
        temperature: float = 1.0,
    ) -> str:
        """Generate caption for single image"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,  # type: ignore
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()

        return caption

    def unload(self):
        """Free GPU memory"""
        if hasattr(self, "model"):
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class VQAPipeline:
    """Visual Question Answering pipeline (LLaVA or Qwen-VL)"""

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.cache = get_shared_cache()
        self.config = get_config()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32

        cache_dir = self.cache.get_path("MODELS_VLM")

        print(f"Loading VQA model from {model_name}...")
        # TODO: Implement based on chosen model (LLaVA or Qwen-VL)
        # For now, placeholder
        self.processor = None
        self.model = None
        print(f"⚠️  VQA pipeline placeholder - implement in Phase 1b")

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 100,
    ) -> str:
        """Answer question about image"""
        # TODO: Implement
        return f"[VQA placeholder] Question: {question}"

    def unload(self):
        """Free GPU memory"""
        if hasattr(self, "model") and self.model:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Singleton instances (lazy load)
_caption_pipeline: Optional[CaptionPipeline] = None
_vqa_pipeline: Optional[VQAPipeline] = None


def get_caption_pipeline() -> CaptionPipeline:
    """Get or create caption pipeline instance"""
    global _caption_pipeline
    if _caption_pipeline is None:
        _caption_pipeline = CaptionPipeline()
    return _caption_pipeline


def get_vqa_pipeline() -> VQAPipeline:
    """Get or create VQA pipeline instance"""
    global _vqa_pipeline
    if _vqa_pipeline is None:
        _vqa_pipeline = VQAPipeline()
    return _vqa_pipeline
