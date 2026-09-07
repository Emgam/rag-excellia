"""Optional local vision captioning for diagrams/screenshots (off by
default — set ENABLE_VISION_CAPTION=true and install a local model such
as moondream/BLIP to activate). Left as an explicit extension point so
the OCR fallback path (multimodal/ocr) remains the default, dependency-
light behavior described in the README."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def caption_image(path: Path) -> Optional[str]:
    # Intentionally unimplemented by default — see module docstring.
    return None
