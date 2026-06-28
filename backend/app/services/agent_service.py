from typing import Any, Dict, Optional

from app.schemas.chat import ChatMessage, ChatRequest
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.utils.image_processing import remove_background


class AgentService:
    def __init__(self) -> None:
        self.chat_service = ChatService()
        self.rag_service = RAGService()
        self.available_tools = {
            "background_removal": self._tool_remove_bg,
            "remove_bg": self._tool_remove_bg,
            "knowledge_retrieval": self._tool_search_knowledge,
            "visual_analysis": self._tool_visual_analysis,
        }

    async def run_reasoning_loop(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if "knowledge" in user_input.lower() or "rag" in user_input.lower():
            result = await self.execute_task("knowledge_retrieval", {"query": user_input})
            return {"content": "I searched the demo knowledge base and found relevant project context.", "tool_used": "knowledge_retrieval", "tool_result": result}
        response = await self.chat_service.generate_response(ChatRequest(messages=[ChatMessage(role="user", content=user_input)]))
        return {"content": response.message.content, "tool_used": None}

    async def execute_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if task_type not in self.available_tools:
            raise ValueError(f"Unknown tool: {task_type}")
        return await self.available_tools[task_type](data)

    async def _tool_remove_bg(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_bytes = data.get("image_bytes")
        if not image_bytes:
            return {"error": "Missing image_bytes"}
        output_bytes = remove_background(image_bytes)
        return {"status": "success", "message": "Background processed", "size": len(output_bytes), "output_path": "demo://processed/background.png"}

    async def _tool_search_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"results": self.rag_service.query(data.get("query", ""), k=2)}

    async def _tool_visual_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "description": "A demo visual asset with clear foreground structure and portfolio-ready metadata.",
            "objects": ["interface", "image-panel", "semantic-tags"],
        }
