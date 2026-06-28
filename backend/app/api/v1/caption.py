import io
import time

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from app.schemas.caption import CaptionResponse

router = APIRouter(prefix="/caption", tags=["Vision - Caption"])


def _avg_rgb_hex(image: Image.Image) -> str:
    im = image.convert("RGB").resize((64, 64))
    pixels = list(im.getdata())
    r = sum(p[0] for p in pixels) / len(pixels)
    g = sum(p[1] for p in pixels) / len(pixels)
    b = sum(p[2] for p in pixels) / len(pixels)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


@router.post("/", response_model=CaptionResponse)
async def generate_caption(
    file: UploadFile = File(...),
    max_length: int = Query(50, ge=10, le=200),
    num_beams: int = Query(3, ge=1, le=10),
    temperature: float = Query(1.0, ge=0.1, le=2.0),
) -> CaptionResponse:
    start = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.verify()
        image = Image.open(io.BytesIO(await file.seek(0) or await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    width, height = image.size
    fmt = (image.format or "unknown").upper()
    avg = _avg_rgb_hex(image)
    caption = (
        f"Demo caption: {fmt} image, {width}x{height}px, average color {avg}. "
        f"Parameters max_length={max_length}, beams={num_beams}, temperature={temperature:.1f}."
    )
    model = "demo:pillow-metadata"
    return CaptionResponse(
        caption=caption,
        confidence=0.74,
        safety_score=0.99,
        is_safe=True,
        model=model,
        model_used=model,
        processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
    )
