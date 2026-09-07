"""Local CPU OCR (Tesseract) for images and, optionally, scanned PDF pages.
Off by default cost-wise (pure CPU, no network) so it's safe to leave on."""
from __future__ import annotations

from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def ocr_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("pytesseract/Pillow not installed; skipping OCR for %s", path)
        return ""

    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed for %s: %s", path, exc)
        return ""
