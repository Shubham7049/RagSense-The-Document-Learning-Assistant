"""Custom exceptions hierarchy for RAG Sense."""


class RAGSenseError(Exception):
    """Base exception for all RAG Sense errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(RAGSenseError):
    """Raised when application configuration is invalid or missing."""


class AppInitializationError(RAGSenseError):
    """Raised when the application fails during startup or component initialization."""


# ==============================================================================
# Domain Exception Stubs (for upcoming phases)
# ==============================================================================


class AuthenticationError(RAGSenseError):
    """Raised for user authentication, registration, or session failures."""


class DocumentProcessingError(RAGSenseError):
    """Raised when document loading, parsing, or chunking fails."""


class VectorStoreError(RAGSenseError):
    """Raised for ChromaDB, embedding generation, or retrieval failures."""


class LLMError(RAGSenseError):
    """Raised for Groq API, prompt formatting, or response streaming errors."""


class ChatHistoryError(RAGSenseError):
    """Raised for chat session or message persistence/retrieval failures."""
