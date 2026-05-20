"""
utils/logger.py — structured console logger using Rich
"""
from __future__ import annotations

import logging
import sys
from rich.logging import RichHandler
from config import cfg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    handler = RichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
