"""Logging utility for RAG Sense."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import settings


def setup_logger(name: str = "rag_sense") -> logging.Logger:
    """Configure and return a structured logger with console and file handlers."""
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    # Resolve log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Log format
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (logs/rag_sense.log)
    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = settings.logs_dir / "rag_sense.log"
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logger in {settings.logs_dir}: {e}")

    return logger


# Default application logger instance
logger = setup_logger("rag_sense")
