"""Recursive file discovery. No manual registration needed — drop files
into DOCUMENTS_DIR (any depth) and they will be found on the next /ingest."""
from __future__ import annotations

from pathlib import Path
from typing import List, Set

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

_IGNORE_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".DS_Store"}


def _get_extensions() -> Set[str]:
    settings = get_settings()
    ext_list = settings.SUPPORTED_EXTENSIONS
    logger.info(f"Loaded {len(ext_list)} supported extensions from settings")
    return set(ext_list)


def discover_files(root: Path) -> List[Path]:
    logger.info(f"discover_files: root={root}, exists={root.exists()}")
    if not root.exists():
        return []
    
    extensions = _get_extensions()
    logger.info(f"extensions: {extensions}")
    
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            logger.debug(f"Checking {p}")
            if p.suffix.lower() in extensions:
                files.append(p)
                logger.debug(f"Added {p}")
    logger.info(f"Found {len(files)} files")
    return files