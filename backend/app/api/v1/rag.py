from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Dict, Any, List
from app.services.rag_service import RAGService
from app.services.agent_service import AgentService
from pydantic import BaseModel
import json

router = APIRouter()
rag_service = RAGService()
agent_service = AgentService()

class StructuredDataRequest(BaseModel):
    name: str
    category: str
    attributes: Dict[str, Any]
    tags: List[str]
    description: str

@router.post("/ingest/structured")
async def ingest_structured(request: StructuredDataRequest):
    """Ingest structured data from form"""
    num_splits = rag_service.ingest_structured_data(request.model_dump())
    return {"status": "success", "num_splits": num_splits}

@router.post("/ingest/markdown")
async def ingest_markdown(text: str = Form(...), metadata_json: str = Form(None)):
    """Ingest raw markdown text"""
    metadata = json.loads(metadata_json) if metadata_json else None
    num_splits = rag_service.ingest_markdown(text, metadata)
    return {"status": "success", "num_splits": num_splits}

@router.get("/query")
async def query_rag(q: str, k: int = 4):
    """Query the RAG system"""
    results = rag_service.query(q, k)
    return {"results": results}

@router.post("/ingest/vlm")
async def ingest_vlm(file: UploadFile = File(...)):
    """
    VLM Image Transformation: Upload image, analyze with Agent,
    and ingest results as Markdown.
    """
    contents = await file.read()

    # Trigger Agent visual analysis
    agent_result = await agent_service.execute_task("visual_analysis", {
        "image_bytes": contents,
        "name": file.filename
    })

    # The agent_result now contains the description and objects
    analysis_result = {
        "name": file.filename,
        "category": "Visual Ingestion",
        "attributes": {"objects": agent_result.get("objects", [])},
        "tags": ["vlm-auto", "vision"],
        "description": agent_result.get("description", "")
    }

    num_splits = rag_service.ingest_structured_data(analysis_result)
    return {"status": "success", "analysis": analysis_result, "num_splits": num_splits}
