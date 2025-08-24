# backend/app/utils/image_processing.py
from PIL import Image, ImageFilter
import cv2
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

logger = logging.getLogger(__name__)


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (384, 384),
    maintain_aspect_ratio: bool = True,
) -> Image.Image:
    """
    Preprocess image for model input
    """
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize with aspect ratio preservation
        if maintain_aspect_ratio:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)

            # Create new image with target size and paste centered
            new_image = Image.new("RGB", target_size, (255, 255, 255))
            paste_x = (target_size[0] - image.width) // 2
            paste_y = (target_size[1] - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        else:
            image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image

    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise


def blur_faces(image: Image.Image, blur_strength: int = 15) -> Image.Image:
    """
    Blur detected faces in the image (optional safety feature)
    """
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Load face cascade (you'd need to download this)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
