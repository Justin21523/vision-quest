from typing import Any, Dict, List, Optional


class RAGService:
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = [
            {
                "content": "VisionQuest combines captioning, VQA, chat, RAG, agents, and a text adventure in one FastAPI API.",
                "metadata": {"source": "demo_seed", "name": "platform-overview"},
            },
            {
                "content": "Mock-safe mode keeps the portfolio demo usable without GPU, model weights, PostgreSQL, or external APIs.",
                "metadata": {"source": "demo_seed", "name": "mock-safe-mode"},
            },
            {
                "content": "The React interface is organized into product modules: vision, knowledge, adventure, agent, history, chat, and telemetry.",
                "metadata": {"source": "demo_seed", "name": "frontend-modules"},
            },
        ]

    def ingest_markdown(self, markdown_text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        chunks = [chunk.strip() for chunk in markdown_text.split("\n\n") if chunk.strip()]
        for index, chunk in enumerate(chunks or [markdown_text]):
            self.documents.append({"content": chunk, "metadata": {"chunk": index, **(metadata or {})}})
        return len(chunks or [markdown_text])

    def ingest_structured_data(self, data: Dict[str, Any]) -> int:
        name = data.get("name") or "Untitled"
        category = data.get("category") or "General"
        description = data.get("description") or ""
        tags = ", ".join(data.get("tags") or [])
        attributes = data.get("attributes") or {}
        markdown = f"# {name}\n\nCategory: {category}\nTags: {tags}\nAttributes: {attributes}\n\n{description}"
        return self.ingest_markdown(markdown, {"source": "structured_form", "name": name, "category": category})

    def query(self, question: str, k: int = 4) -> List[Dict[str, Any]]:
        words = {part.lower() for part in question.split() if len(part) > 2}

        def score(doc: Dict[str, Any]) -> int:
            content = doc["content"].lower()
            return sum(1 for word in words if word in content)

        ranked = sorted(self.documents, key=score, reverse=True)
        return ranked[: max(1, k)]
