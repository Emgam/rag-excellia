import logging
import sys

from config.settings import get_settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    settings = get_settings()
    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
