from typing import List, Iterator, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ..schemas.chat import ChatMessage, ChatRequest, ChatResponse
from ..core.config import settings


class ChatService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = settings.DEFAULT_LLM_MODEL

    def load_model(self, model_name: Optional[str] = None):
        """載入 LLM 模型"""
        if model_name:
            self.model_name = model_name

        if self.model_name == "qwen":
            model_path = "Qwen/Qwen2-7B-Instruct"
        elif self.model_name == "llama":
            model_path = "meta-llama/Llama-3.1-8B-Instruct"
        else:
            model_path = self.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """格式化對話為 prompt"""
        if self.model_name == "qwen":
            return self.tokenizer.apply_chat_template(
                [{"role": msg.role, "content": msg.content} for msg in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:  # llama 格式
            formatted = "<|begin_of_text|>"
            for msg in messages:
                formatted += f"<|start_header_id|>{msg.role}<|end_header_id|>\n{msg.content}<|eot_id|>"
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n"
            return formatted

    def generate_response(self, request: ChatRequest) -> ChatResponse:
        """生成對話回應"""
        if not self.model:
            self.load_model(request.model)

        # 格式化輸入
        prompt = self.format_messages(request.messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成回應
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解碼回應
        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text.strip()),
            model=self.model_name,
            usage={"total_tokens": len(outputs[0])},
        )

    def generate_stream(self, request: ChatRequest) -> Iterator[str]:
        """串流生成回應"""
        # 簡化版串流實作
        response = self.generate_response(request)
        words = response.message.content.split()

        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield f" {word}"
