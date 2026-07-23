"""
logger.py
Shared logger setup used across the pipeline. Every script calls
get_logger(__name__) instead of using print(), so pipeline runs are
timestamped, leveled (INFO/WARNING/ERROR), and auditable after the fact —
useful the moment this runs unattended (e.g. a scheduled retraining job)
rather than watched in a terminal.
"""

import logging
import sys

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers if get_logger is called more than
        # once for the same module (e.g. re-imported in a notebook).
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    return logger
