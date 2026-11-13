"""Structured logging via loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config import settings

_configured = False


def get_logger(name: str):
    global _configured
    if not _configured:
        logger.remove()
        logger.add(sys.stderr, level="INFO",
                   format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
                          "<cyan>{name}</cyan> | {message}")
        log_path: Path = settings.logs_dir / "monitor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level="DEBUG", rotation="10 MB", retention=5)
        _configured = True
    return logger.bind(name=name)
