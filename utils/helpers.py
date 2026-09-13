"""Helper utility functions for RAG Sense."""

import platform
import sys
from pathlib import Path
from typing import Any

from config.settings import settings
from core.exceptions import ConfigurationError


def ensure_directory(path: Path) -> Path:
    """Ensure that a directory exists, creating parents if necessary."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise ConfigurationError(f"Failed to create directory at '{path}': {e}") from e


def init_project_directories() -> dict[str, Path]:
    """Initialize essential directories for data and logs."""
    return {
        "data_dir": ensure_directory(settings.data_dir),
        "logs_dir": ensure_directory(settings.logs_dir),
    }


def get_system_info() -> dict[str, Any]:
    """Return system and runtime information for diagnostic and UI displays."""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "app_env": settings.app_env,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "base_dir": str(settings.base_dir),
        "data_dir": str(settings.data_dir),
        "logs_dir": str(settings.logs_dir),
    }


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    """Mask a secret string for safe display in logs or UI."""
    if not value:
        return "Not configured"
    if len(value) <= visible_chars:
        return "***"
    return f"{value[:visible_chars]}...{value[-visible_chars:]}"
