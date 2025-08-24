# backend/app/services/vqa_service.py
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.utils.image_processing import preprocess_image
from backend.app.utils.safety import content_filter
import logging

logger = logging.getLogger(__name__)


class VQAService:
    def __init__(self):
        self.processor = None
        self.model = None
        self.current_model = "llava"  # or "qwen-vl"
        self._load_model()

    def _load_model(self):
        """Load LLaVA or Qwen-VL model for VQA"""
        try:
            if self.current_model == "llava":
                model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

                self.processor = LlavaNextProcessor.from_pretrained(model_name)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            elif self.current_model == "qwen-vl":
                # Alternative: Qwen-VL for better Chinese support
                model_name = "Qwen/Qwen-VL-Chat"
                # Implementation would go here

            logger.info(f"VQA model {self.current_model} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VQA model: {e}")
            raise

    async def answer_question(
        self,
        image: Image.Image,
        question: str,
        language: str = "en",
        max_length: int = 100,
    ) -> dict:
        """
        Answer visual questions using LLaVA/Qwen-VL
        """
        try:
            # Preprocess image
            processed_image = preprocess_image(image, target_size=(336, 336))

            # Format conversation for LLaVA
            if language.startswith("zh"):
                # Chinese prompt
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"請回答這個關於圖片的問題：{question}",
                            },
                            {"type": "image", "image": processed_image},
                        ],
                    },
                ]
            else:
                # English prompt
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Answer this question about the image: {question}",
                            },
                            {"type": "image", "image": processed_image},
                        ],
                    },
                ]

            # Apply chat template
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            # Prepare inputs
            inputs = self.processor(prompt, processed_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate answer
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Extract only the new tokens (answer)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)

            # Apply content filter
            filtered_answer = content_filter(answer)

            return {
                "answer": filtered_answer,
                "raw_answer": answer,
                "question": question,
                "language": language,
                "confidence": 0.82,  # Placeholder
                "model_used": self.current_model,
            }

        except Exception as e:
            logger.error(f"VQA error: {e}")
            raise
