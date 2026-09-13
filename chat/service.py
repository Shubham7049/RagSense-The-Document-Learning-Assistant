"""Chat service orchestrating multi-chat threads, persistence, and history context."""

import re
import uuid
from pathlib import Path

from chat.history import (
    delete_chat_vectors,
    format_messages_for_prompt,
    index_chat_message,
    retrieve_relevant_older_history,
)
from chat.models import ChatMessage, ChatThread
from chat.repository import ChatRepository
from config.settings import settings
from core.exceptions import RAGSenseError
from utils.logger import logger


class ChatService:
    """Coordinates persistent chat thread lifecycle, messages, and contextual history retrieval."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        persist_dir: Path | str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.repository = ChatRepository(db_path=db_path)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def create_new_chat(self, user_id: int, title: str = "New Chat") -> ChatThread:
        """Create a new chat thread for the specified user."""
        chat_id = str(uuid.uuid4())
        return self.repository.create_chat(chat_id=chat_id, user_id=user_id, title=title)

    def list_user_chats(self, user_id: int) -> list[ChatThread]:
        """List all chat threads belonging to a user, ordered by most recent activity."""
        return self.repository.list_chats(user_id=user_id)

    def get_chat(self, chat_id: str, user_id: int) -> ChatThread | None:
        """Retrieve a specific chat thread if owned by user_id."""
        return self.repository.get_chat(chat_id=chat_id, user_id=user_id)

    def rename_chat(self, chat_id: str, user_id: int, new_title: str) -> bool:
        """Rename a chat thread, ensuring ownership validation."""
        clean_title = new_title.strip()
        if not clean_title:
            raise ValueError("Chat title cannot be empty.")
        return self.repository.update_chat_title(chat_id=chat_id, user_id=user_id, title=clean_title)

    def delete_chat(self, chat_id: str, user_id: int) -> bool:
        """Delete a chat thread, all its SQLite messages, and its ChromaDB vector history."""
        # Verify ownership
        chat = self.get_chat(chat_id=chat_id, user_id=user_id)
        if not chat:
            logger.warning(f"Unauthorized or non-existent chat deletion attempt: {chat_id} (user #{user_id})")
            return False

        deleted = self.repository.delete_chat(chat_id=chat_id, user_id=user_id)
        if deleted:
            delete_chat_vectors(
                chat_id=chat_id,
                user_id=user_id,
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )
        return deleted

    def add_message(
        self,
        chat_id: str,
        user_id: int,
        role: str,
        content: str,
    ) -> ChatMessage:
        """Store a message in SQLite and index it into vector history for semantic search."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Message content cannot be empty.")

        # Ensure chat exists and belongs to user
        chat = self.get_chat(chat_id=chat_id, user_id=user_id)
        if not chat:
            raise RAGSenseError(f"Chat thread '{chat_id}' does not exist or does not belong to user #{user_id}.")

        message_id = str(uuid.uuid4())
        msg = self.repository.create_message(
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=clean_content,
        )

        # Index in vector store for older history retrieval
        index_chat_message(
            message=msg,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )
        return msg

    def get_chat_messages(self, chat_id: str, user_id: int) -> list[ChatMessage]:
        """Fetch all messages for a chat, verifying ownership."""
        chat = self.get_chat(chat_id=chat_id, user_id=user_id)
        if not chat:
            return []
        return self.repository.get_messages(chat_id=chat_id, user_id=user_id)

    def get_contextual_history(
        self,
        chat_id: str,
        user_id: int,
        query: str,
        recent_k: int | None = None,
        older_k: int | None = None,
    ) -> tuple[str, str]:
        """Retrieve (recent_history, relevant_older_history) for LLM context.

        Returns
        -------
        recent_history_str : str
            Immediate prior dialog turns (last recent_k messages).
        relevant_older_history_str : str
            Semantically relevant older messages retrieved via vector search, excluding recent turns.
        """
        k_recent = recent_k or settings.recent_history_k
        k_older = older_k or settings.relevant_history_k

        # 1. Fetch recent messages
        recent_msgs = self.repository.get_recent_messages(chat_id=chat_id, user_id=user_id, limit=k_recent)
        recent_str = format_messages_for_prompt(recent_msgs)

        # 2. Exclude recent message IDs from older retrieval
        exclude_ids = {m.id for m in recent_msgs}

        # 3. Retrieve semantically relevant older messages if conversation is deep
        older_docs = retrieve_relevant_older_history(
            user_id=user_id,
            chat_id=chat_id,
            query=query,
            exclude_message_ids=exclude_ids,
            k=k_older,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )
        older_str = format_messages_for_prompt(older_docs)

        return recent_str, older_str

    @staticmethod
    def generate_title_from_prompt(prompt: str, max_chars: int = 35) -> str:
        """Derive a clean, concise title from the user's initial prompt without extra LLM overhead."""
        # Strip code blocks, markdown, multiple spaces
        cleaned = re.sub(r"[`#*_\r\n]+", " ", prompt).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        if not cleaned:
            return "New Chat"

        # Split on sentence enders (avoid splitting on decimal numbers like 3.12)
        parts = re.split(r"(?<=[?!])\s+|(?<=\.)\s+", cleaned)
        candidate = parts[0].strip() if parts else cleaned

        if len(candidate) > max_chars:
            candidate = candidate[:max_chars].rstrip() + "..."

        # Title case first letter
        return candidate[0].upper() + candidate[1:]
