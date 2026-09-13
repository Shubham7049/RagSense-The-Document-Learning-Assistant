"""Tests for centralized configuration, utilities, and exceptions in RAG Sense."""

import pytest
from pydantic import ValidationError

from config.settings import Settings, get_settings
from core.exceptions import ConfigurationError, RAGSenseError
from utils.helpers import ensure_directory, get_system_info, mask_secret
from utils.logger import setup_logger


def test_default_settings():
    """Test that Settings loads default development values correctly."""
    # Force new instance without depending on cached singleton
    settings = Settings(_env_file=None)

    assert settings.app_name == "RAG Sense"
    assert settings.app_env == "development"
    assert settings.debug is True
    assert settings.log_level == "INFO"
    assert settings.app_version == "0.1.0"
    assert settings.max_upload_size_mb == 100
    assert settings.base_dir.exists()
    assert settings.data_dir.name == "data"
    assert settings.logs_dir.name == "logs"
    assert settings.database_path.name == "ragsense.db"


def test_settings_env_override(monkeypatch):
    """Test that environment variables cleanly override default settings."""
    monkeypatch.setenv("APP_NAME", "Test Sense")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("APP_VERSION", "1.2.3")
    monkeypatch.setenv("DATABASE_PATH", "custom_data/custom.db")

    overridden = Settings(_env_file=None)

    assert overridden.app_name == "Test Sense"
    assert overridden.app_env == "production"
    assert overridden.debug is False
    assert overridden.log_level == "WARNING"
    assert overridden.app_version == "1.2.3"
    assert overridden.database_path.name == "custom.db"


def test_invalid_log_level(monkeypatch):
    """Test that an invalid LOG_LEVEL raises a validation error."""
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE_INVALID")
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    assert "Invalid LOG_LEVEL" in str(exc_info.value)


def test_invalid_app_env(monkeypatch):
    """Test that an invalid APP_ENV raises a validation error."""
    monkeypatch.setenv("APP_ENV", "local_sandbox")
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    assert "Invalid APP_ENV" in str(exc_info.value)


def test_get_settings_caching():
    """Test that get_settings() returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_logger_setup():
    """Test that logger setup initializes properly."""
    log = setup_logger("test_logger")
    assert log.name == "test_logger"
    assert len(log.handlers) > 0


def test_helpers(tmp_path):
    """Test helper functions for directories, masking, and system info."""
    test_dir = tmp_path / "test_subdir"
    created = ensure_directory(test_dir)
    assert created.exists()

    masked = mask_secret("gsk_1234567890abcdef", visible_chars=4)
    assert masked.startswith("gsk_")
    assert masked.endswith("cdef")
    assert "..." in masked

    assert mask_secret(None) == "Not configured"
    assert mask_secret("abc", visible_chars=4) == "***"

    info = get_system_info()
    assert "app_name" in info
    assert "python_version" in info
    assert "platform" in info


def test_exceptions():
    """Test custom exception hierarchy."""
    err = ConfigurationError("Config failed", details={"key": "LOG_LEVEL"})
    assert isinstance(err, RAGSenseError)
    assert "Config failed" in str(err)
    assert err.details == {"key": "LOG_LEVEL"}
