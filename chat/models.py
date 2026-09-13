"""Chat models for persistent threads and messages."""

from enum import Enum
from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """Permitted message sender roles."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatThread(BaseModel):
    """Represents a persistent conversation thread belonging to a specific user."""

    id: str = Field(description="Unique chat thread UUID")
    user_id: int = Field(description="Owning user ID for strict isolation")
    title: str = Field(default="New Chat", description="Human-readable chat title")
    created_at: str = Field(description="ISO timestamp of chat creation")
    updated_at: str = Field(description="ISO timestamp of last activity in chat")


class ChatMessage(BaseModel):
    """Represents an individual persistent message within a chat thread."""

    id: str = Field(description="Unique message UUID")
    chat_id: str = Field(description="Parent chat thread ID")
    user_id: int = Field(description="Owning user ID for verification")
    role: str = Field(description="Role: 'user' or 'assistant'")
    content: str = Field(description="Message body text")
    created_at: str = Field(description="ISO timestamp of message creation")
