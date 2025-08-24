# backend/app/utils/safety.py
import re
from typing import List, Optional
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Basic content filters (extend as needed)
NSFW_KEYWORDS = [
    "explicit",
    "sexual",
    "nude",
    "naked",
    "pornographic",
    "inappropriate",
    "adult content",
]

SENSITIVE_TERMS = ["violence", "weapon", "drug", "illegal", "harmful"]


def content_filter(text: str, strict_mode: bool = False) -> str:
    """
    Filter potentially sensitive content from generated text
    """
    if not settings.ENABLE_CONTENT_FILTER:
        return text

    try:
        filtered_text = text.strip()

        # Basic NSFW filtering
        for keyword in NSFW_KEYWORDS:
            if keyword.lower() in filtered_text.lower():
                if strict_mode:
                    return "[Content filtered for safety]"
                filtered_text = re.sub(
                    re.escape(keyword), "[filtered]", filtered_text, flags=re.IGNORECASE
                )

        # Sensitive terms (warning only in logs)
        for term in SENSITIVE_TERMS:
            if term.lower() in filtered_text.lower():
                logger.warning(f"Sensitive content detected: {term}")

        return filtered_text

    except Exception as e:
        logger.error(f"Content filtering error: {e}")
        return text  # Return original if filtering fails


def validate_prompt(prompt: str, max_length: int = 500) -> bool:
    """
    Validate user prompts for safety and length
    """
    if len(prompt) > max_length:
        return False

    # Add more validation rules as needed
    forbidden_patterns = [r"jailbreak", r"ignore.*instruction", r"system.*prompt"]

    for pattern in forbidden_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False

    return True
