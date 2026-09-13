"""Core package for RAG Sense."""

from core.database import get_connection, get_db_context, init_db
from core.exceptions import (
    AppInitializationError,
    AuthenticationError,
    ChatHistoryError,
    ConfigurationError,
    DocumentProcessingError,
    LLMError,
    RAGSenseError,
    VectorStoreError,
)

__all__ = [
    "AppInitializationError",
    "AuthenticationError",
    "ChatHistoryError",
    "ConfigurationError",
    "DocumentProcessingError",
    "LLMError",
    "RAGSenseError",
    "VectorStoreError",
    "get_connection",
    "get_db_context",
    "init_db",
]
