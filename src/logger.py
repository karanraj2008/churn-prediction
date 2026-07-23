"""
logger.py
Shared logger setup used across the pipeline. Every script calls
get_logger(__name__) instead of using print(), so pipeline runs are
timestamped, leveled (INFO/WARNING/ERROR), and auditable after the fact —
useful the moment this runs unattended (e.g. a scheduled retraining job)
rather than watched in a terminal. Logs go to both the console AND a
persistent file (logs/pipeline.log) at the project root.
"""

import logging
import sys
from pathlib import Path

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"


def get_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers if get_logger is called more than
        # once for the same module (e.g. re-imported in a notebook).
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console output — what you see while the script runs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File output — persisted history of every run, appended to over time
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger