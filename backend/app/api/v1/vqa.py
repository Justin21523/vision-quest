# backend/app/api/v1/vqa.py
"""
VQA API (portfolio demo - lightweight).

Avoids heavyweight ML dependencies (LLaVA/Qwen-VL) while keeping an interactive
demo endpoint for the portfolio site.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from PIL import Image
import io

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
    elif ("size" in q_lower) or ("resolution" in q_lower) or ("多大" in q) or ("尺寸" in q):
        answer = f"Image resolution is {width}×{height}."
    else:
        answer = (
            "This is a lightweight portfolio demo (no large VLM running on the server). "
            f"Image is {width}×{height}, avg_color={avg}."
        )

    return VQAResponse(
        answer=answer,
        raw_answer=answer,
        question=q,
        language=lang,
        confidence=0.25,
        model_used="demo:vqa-rules",
    )
