# backend/app/api/v1/vqa.py
"""
VQA API (portfolio demo - lightweight).

Avoids heavyweight ML dependencies (LLaVA/Qwen-VL) while keeping an interactive
demo endpoint for the portfolio site.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from PIL import Image
import io
import time

from app.schemas.vqa import VQAResponse

router = APIRouter(prefix="/vqa", tags=["Vision - VQA (Demo)"])


def _avg_rgb_hex(image: Image.Image) -> str:
    im = image.convert("RGB")
    im = im.resize((64, 64))
    pixels = list(im.getdata())
    r = sum(p[0] for p in pixels) / len(pixels)
    g = sum(p[1] for p in pixels) / len(pixels)
    b = sum(p[2] for p in pixels) / len(pixels)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


@router.post("/", response_model=VQAResponse)
async def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(...),
    lang: str = Form(default="en"),
    max_length: int = Form(default=100),
):
    start = time.perf_counter()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    width, height = image.size
    avg = _avg_rgb_hex(image)

    q_lower = q.lower()
    if ("color" in q_lower) or ("顏色" in q) or ("颜色" in q):
        answer = f"Average color is {avg} (demo heuristic)."
        reasoning = ["Detected color-related wording in the question.", "Computed an average RGB swatch with Pillow."]
    elif ("size" in q_lower) or ("resolution" in q_lower) or ("多大" in q) or ("尺寸" in q):
        answer = f"Image resolution is {width}×{height}."
        reasoning = ["Detected size or resolution wording.", "Read image dimensions from the uploaded file header."]
    elif ("safe" in q_lower) or ("risk" in q_lower) or ("安全" in q):
        answer = "Use mock-safe review mode: inspect visual evidence, retrieve context, then update state only after confirmation."
        reasoning = ["Detected safety/risk wording.", "Mapped the question to the demo review workflow."]
    else:
        answer = (
            "This is a lightweight portfolio demo (no large VLM running on the server). "
            f"Image is {width}×{height}, avg_color={avg}."
        )
        reasoning = ["Validated the image.", "Returned deterministic metadata so the public demo works without model weights."]

    return VQAResponse(
        answer=answer,
        raw_answer=answer,
        question=q,
        language=lang,
        confidence=0.25,
        model_used="demo:vqa-rules",
        evidence=[
            f"resolution={width}x{height}",
            f"avg_color={avg}",
            "mode=mock-safe",
        ],
        reasoning=reasoning,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
