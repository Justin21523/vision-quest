# backend/app/utils/image_processing.py
from PIL import Image, ImageFilter
import cv2
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path
import sys
import io

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

logger = logging.getLogger(__name__)

# Supported formats
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DIMENSION = 4096


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""

    pass


def validate_image(image: Image.Image) -> None:
    """
    Validate image format and size

    Raises:
        ImageProcessingError: If image is invalid
    """
    # Check format
    if image.format not in SUPPORTED_FORMATS:
        raise ImageProcessingError(
            f"Unsupported format: {image.format}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check dimensions
    width, height = image.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageProcessingError(
            f"Image too large: {width}x{height}. " f"Max dimension: {MAX_DIMENSION}"
        )

    # Check file size (approximate)
    buffer = io.BytesIO()
    image.save(buffer, format=image.format)
    size = buffer.tell()
    if size > MAX_IMAGE_SIZE:
        raise ImageProcessingError(
            f"Image file too large: {size / 1024 / 1024:.1f}MB. "
            f"Max: {MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB"
        )


def preprocess_image(
    image: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    convert_rgb: bool = True,
) -> Image.Image:
    """
    Preprocess image for model input

    Args:
        image: Input PIL Image
        target_size: Optional (width, height) to resize to
        normalize: Whether to normalize pixel values
        convert_rgb: Convert to RGB if needed

    Returns:
        Preprocessed PIL Image
    """
    # Convert to RGB if needed
    if convert_rgb and image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if target size specified
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Note: Normalization typically handled by model processor
    # This is kept as a flag for custom preprocessing

    return image


def resize_keep_aspect(
    image: Image.Image,
    max_size: int = 1024,
) -> Image.Image:
    """
    Resize image while keeping aspect ratio

    Args:
        image: Input PIL Image
        max_size: Maximum dimension (width or height)

    Returns:
        Resized PIL Image
    """
    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def center_crop(
    image: Image.Image,
    size: Tuple[int, int],
) -> Image.Image:
    """
    Center crop image to specified size

    Args:
        image: Input PIL Image
        size: Target (width, height)

    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    target_width, target_height = size

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))


def blur_faces(image: Image.Image, blur_strength: int = 15) -> Image.Image:
    """
    Blur detected faces in the image (optional safety feature)
    """
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Load face cascade (you'd need to download this)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
        )

        # Detect faces
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Blur each face
        for x, y, w, h in faces:
            face_region = cv_image[y : y + h, x : x + w]
            blurred_face = cv2.GaussianBlur(
                face_region, (blur_strength, blur_strength), 0
            )
            cv_image[y : y + h, x : x + w] = blurred_face

        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    except Exception as e:
        logger.warning(f"Face blurring failed, returning original: {e}")
        return image
