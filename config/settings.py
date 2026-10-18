"""Centralized configuration for RAG Sense using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the project (points to rag-sense/)
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Application configuration with validation and sensible defaults."""

    # Core Application Settings
    app_name: str = Field(default="RAG Sense", description="Application display name")
    app_env: str = Field(default="development", description="Environment: development, staging, production, testing")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    app_version: str = Field(default="0.1.0", description="Semantic application version")

    # Project Directories
    base_dir: Path = Field(default=BASE_DIR, description="Root directory of the project")
    data_dir: Path = Field(default=BASE_DIR / "data", description="Directory for persistent data")
    logs_dir: Path = Field(default=BASE_DIR / "logs", description="Directory for log files")

    # Database Configuration
    database_path: Path = Field(
        default=BASE_DIR / "data" / "ragsense.db",
        description="Path to SQLite database file",
    )

    # Document Ingestion & Storage Configuration
    max_upload_size_mb: int = Field(default=100, description="Maximum allowed file size in MB")
    document_storage_directory: Path = Field(
        default=BASE_DIR / "data" / "users",
        description="Path to user-partitioned document storage",
    )
    chunk_size: int = Field(default=1000, description="RecursiveCharacterTextSplitter chunk size")
    chunk_overlap: int = Field(default=200, description="RecursiveCharacterTextSplitter chunk overlap")

    # Vector Store & Embedding Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace / Sentence Transformer model name",
    )
    chroma_persist_directory: Path = Field(
        default=BASE_DIR / "data" / "chroma",
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="rag_sense_docs",
        description="ChromaDB collection name",
    )

    # RAG & Groq LLM Configuration
    groq_api_key: str | None = Field(default=None, description="Groq Cloud API Key")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq LLM model identifier")
    retrieval_k: int = Field(default=4, description="Default number of chunks to retrieve for RAG context")

    # Chat & History Configuration
    recent_history_k: int = Field(default=6, description="Number of recent messages to include as immediate dialog")
    relevant_history_k: int = Field(default=3, description="Number of relevant older messages to retrieve via vector similarity")
    chroma_chat_collection_name: str = Field(default="rag_sense_chat_history", description="ChromaDB collection for chat message history")

    model_config = SettingsConfigDict(
        env_file=ENV_FILE if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def resolve_groq_api_key(cls, v: str | None) -> str | None:
        """Resolve Groq API key, falling back to Streamlit secrets if running in cloud."""
        if v and str(v).strip() and str(v).strip() != "your_groq_api_key_here":
            return str(v).strip()
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if "GROQ_API_KEY" in st.secrets:
                    return str(st.secrets["GROQ_API_KEY"]).strip()
                if "groq_api_key" in st.secrets:
                    return str(st.secrets["groq_api_key"]).strip()
        except Exception:
            pass
        return v

    @field_validator("database_path", mode="before")
    @classmethod
    def resolve_database_path(cls, v: str | Path | None) -> Path:
        """Resolve database path, anchoring relative paths to BASE_DIR."""
        if v is None or str(v).strip() == "":
            return BASE_DIR / "data" / "ragsense.db"
        path = Path(v)
        if not path.is_absolute():
            return BASE_DIR / path
        return path

    @field_validator("document_storage_directory", mode="before")
    @classmethod
    def resolve_storage_dir(cls, v: str | Path | None) -> Path:
        """Resolve document storage directory, anchoring relative paths to BASE_DIR."""
        if v is None or str(v).strip() == "":
            return BASE_DIR / "data" / "users"
        path = Path(v)
        if not path.is_absolute():
            return BASE_DIR / path
        return path

    @field_validator("chroma_persist_directory", mode="before")
    @classmethod
    def resolve_chroma_dir(cls, v: str | Path | None) -> Path:
        """Resolve ChromaDB persistence directory, anchoring relative paths to BASE_DIR."""
        if v is None or str(v).strip() == "":
            return BASE_DIR / "data" / "chroma"
        path = Path(v)
        if not path.is_absolute():
            return BASE_DIR / path
        return path

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a valid logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.strip().upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid LOG_LEVEL '{v}'. Must be one of: {', '.join(sorted(valid_levels))}")
        return upper_v

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Ensure app_env is one of the supported environment names."""
        valid_envs = {"development", "staging", "production", "testing"}
        lower_v = v.strip().lower()
        if lower_v not in valid_envs:
            raise ValueError(f"Invalid APP_ENV '{v}'. Must be one of: {', '.join(sorted(valid_envs))}")
        return lower_v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()


def reload_settings() -> Settings:
    """Clear cached settings and reload from environment / .env."""
    get_settings.cache_clear()
    global settings
    settings = get_settings()
    return settings


# Primary singleton instance for clean, direct import
settings: Settings = get_settings()
