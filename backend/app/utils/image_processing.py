import io

from PIL import Image

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_DIMENSION = 4096


class ImageProcessingError(Exception):
    pass


def remove_background(image_bytes: bytes) -> bytes:
    try:
        from rembg import remove

        return remove(image_bytes)
    except Exception:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def validate_image(image: Image.Image) -> None:
    if image.format and image.format not in SUPPORTED_FORMATS:
        raise ImageProcessingError(f"Unsupported format: {image.format}")
    width, height = image.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageProcessingError(f"Image too large: {width}x{height}")
    buffer = io.BytesIO()
    image.save(buffer, format=image.format or "PNG")
    if buffer.tell() > MAX_IMAGE_SIZE:
        raise ImageProcessingError("Image file too large")
