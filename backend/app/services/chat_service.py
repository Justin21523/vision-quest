import json
import re
import time
from typing import Any, Dict, Iterator, Optional

from app.core.config import settings
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse


class ChatService:
    def __init__(self) -> None:
        self.model_name = settings.DEFAULT_LLM_MODEL

    async def generate_response(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        latest = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        response_text = self._demo_response(latest)
        delta = self._extract_state_delta(response_text)
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text),
            model=self.model_name,
            model_used=self.model_name,
            tokens_used=max(24, len(response_text.split()) + len(latest.split())),
            finish_reason="stop",
            usage={"total_tokens": max(24, len(response_text.split()) + len(latest.split()))},
            state_delta=delta,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    def generate_stream(self, request: ChatRequest) -> Iterator[str]:
        latest = request.messages[-1].content if request.messages else ""
        for token in self._demo_response(latest).split():
            yield token + " "

    def _demo_response(self, prompt: str) -> str:
        lower = prompt.lower()
        if "name" in lower or "名字" in prompt:
            return "我記得你前面提到的名字，並會在多輪對話中保留這個上下文。"
        if "health" in lower or "狀態" in prompt:
            return (
                "The adventure state changed after your action. "
                '```json {"state_delta": {"target": "player", "action": "update", "value": {"health": 85}}} ```'
            )
        if "architecture" in lower or "架構" in prompt:
            return "VisionQuest uses a FastAPI service layer, React demo UI, mock-safe model adapters, and optional pgvector storage."
        return (
            "VisionQuest demo response: I can explain the multimodal pipeline, simulate model output, "
            "and keep the same API shape used by the GPU-backed version."
        )

    def _extract_state_delta(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        return payload.get("state_delta")
