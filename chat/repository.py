"""SQLite repository for persistent chat threads and messages with user isolation."""

from datetime import datetime, timezone
from pathlib import Path

from chat.models import ChatMessage, ChatThread
from core.database import get_db_context
from utils.logger import logger


class ChatRepository:
    """Handles SQLite persistence for chat threads and messages, strictly enforcing per-user isolation."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = db_path

    def create_chat(
        self,
        chat_id: str,
        user_id: int,
        title: str = "New Chat",
    ) -> ChatThread:
        """Insert a new chat thread into SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO chats (id, user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?);
        """
        with get_db_context(self.db_path) as conn:
            conn.execute(query, (chat_id, user_id, title, now, now))

        logger.info(f"Created chat thread '{title}' ({chat_id}) for user #{user_id}")
        return ChatThread(
            id=chat_id,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )

    def get_chat(self, chat_id: str, user_id: int) -> ChatThread | None:
        """Retrieve a specific chat thread only if owned by user_id."""
        query = "SELECT * FROM chats WHERE id = ? AND user_id = ?;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, (chat_id, user_id))
            row = cursor.fetchone()

        if not row:
            return None

        return ChatThread(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_chats(self, user_id: int) -> list[ChatThread]:
        """List all chat threads for a user, ordered by most recent activity."""
        query = "SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, (user_id,))
            rows = cursor.fetchall()

        return [
            ChatThread(
                id=r["id"],
                user_id=r["user_id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def update_chat_title(self, chat_id: str, user_id: int, title: str) -> bool:
        """Update a chat thread's title, strictly verifying ownership."""
        now = datetime.now(timezone.utc).isoformat()
        query = "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, (title.strip(), now, chat_id, user_id))
            return cursor.rowcount > 0

    def touch_chat(self, chat_id: str, user_id: int) -> bool:
        """Update the updated_at timestamp of a chat thread."""
        now = datetime.now(timezone.utc).isoformat()
        query = "UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, (now, chat_id, user_id))
            return cursor.rowcount > 0

    def delete_chat(self, chat_id: str, user_id: int) -> bool:
        """Delete a chat thread and all its messages, verifying ownership."""
        query = "DELETE FROM chats WHERE id = ? AND user_id = ?;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, (chat_id, user_id))
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted chat thread {chat_id} for user #{user_id}")
        return deleted

    def create_message(
        self,
        message_id: str,
        chat_id: str,
        user_id: int,
        role: str,
        content: str,
    ) -> ChatMessage:
        """Insert a persistent message and update parent chat timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO messages (id, chat_id, user_id, role, content, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with get_db_context(self.db_path) as conn:
            conn.execute(query, (message_id, chat_id, user_id, role, content, now))
            # Touch chat updated_at
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?;", (now, chat_id, user_id))

        logger.debug(f"Saved {role} message ({message_id}) in chat {chat_id} for user #{user_id}")
        return ChatMessage(
            id=message_id,
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=content,
            created_at=now,
        )

    def get_messages(
        self,
        chat_id: str,
        user_id: int,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Fetch all messages for a given chat in chronological order."""
        if limit:
            query = """
            SELECT * FROM messages
            WHERE chat_id = ? AND user_id = ?
            ORDER BY created_at ASC
            LIMIT ?;
            """
            params = (chat_id, user_id, limit)
        else:
            query = """
            SELECT * FROM messages
            WHERE chat_id = ? AND user_id = ?
            ORDER BY created_at ASC;
            """
            params = (chat_id, user_id)

        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            ChatMessage(
                id=r["id"],
                chat_id=r["chat_id"],
                user_id=r["user_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_recent_messages(
        self,
        chat_id: str,
        user_id: int,
        limit: int = 6,
    ) -> list[ChatMessage]:
        """Fetch the most recent N messages in chronological order."""
        query = """
        SELECT * FROM messages
        WHERE chat_id = ? AND user_id = ?
        ORDER BY created_at DESC
        LIMIT ?;
        """
        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, (chat_id, user_id, limit))
            rows = cursor.fetchall()

        # Reverse to maintain chronological order
        chronological = list(reversed(rows))
        return [
            ChatMessage(
                id=r["id"],
                chat_id=r["chat_id"],
                user_id=r["user_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in chronological
        ]

    def count_messages(self, chat_id: str, user_id: int) -> int:
        """Count total messages in a chat thread."""
        query = "SELECT COUNT(*) as count FROM messages WHERE chat_id = ? AND user_id = ?;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.execute(query, (chat_id, user_id))
            row = cursor.fetchone()
            return row["count"] if row else 0
