"""Document domain and data models for RAG Sense."""

from enum import Enum
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document ingestion processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    """Represents a stored and tracked user document."""

    id: str = Field(description="Unique document UUID")
    user_id: int = Field(description="Owning user ID for strict isolation")
    filename: str = Field(description="Original filename")
    file_type: str = Field(description="File extension without dot (pdf, docx, etc.)")
    file_size: int = Field(description="Size in bytes")
    storage_path: str = Field(description="Absolute path to stored file on disk")
    upload_timestamp: str = Field(description="ISO-formatted upload timestamp")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Current processing status")
    chunk_count: int = Field(default=0, description="Total chunks generated and indexed")
    error_message: str | None = Field(default=None, description="Failure details if status is FAILED")
