"""Database repository for user operations in SQLite."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from auth.models import User, UserInDB
from core.database import get_db_context
from core.exceptions import AuthenticationError
from utils.logger import logger


class UserRepository:
    """Handles persistent SQLite operations for users."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = db_path

    def create_user(self, username: str, password_hash: str) -> User:
        """Create a new user record in SQLite using parameterized SQL."""
        now = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO users (username, password_hash, created_at)
        VALUES (?, ?, ?);
        """

        try:
            with get_db_context(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (username, password_hash, now))
                user_id = cursor.lastrowid
                logger.info(f"Created user record #{user_id} for username: '{username}'")
                return User(id=user_id, username=username, created_at=now)
        except sqlite3.IntegrityError as e:
            logger.warning(f"Registration conflict: username '{username}' already exists.")
            raise AuthenticationError(f"A user with username '{username}' already exists.") from e

    def get_by_username(self, username: str) -> UserInDB | None:
        """Retrieve user with password hash by username."""
        query = """
        SELECT id, username, password_hash, created_at
        FROM users
        WHERE username = ?;
        """
        with get_db_context(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (username,))
            row = cursor.fetchone()
            if not row:
                return None
            return UserInDB(
                id=row["id"],
                username=row["username"],
                password_hash=row["password_hash"],
                created_at=row["created_at"],
            )

    def get_by_id(self, user_id: int) -> User | None:
        """Retrieve public user profile by ID."""
        query = """
        SELECT id, username, created_at
        FROM users
        WHERE id = ?;
        """
        with get_db_context(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return User(
                id=row["id"],
                username=row["username"],
                created_at=row["created_at"],
            )

    def count_users(self) -> int:
        """Return total count of registered users."""
        query = "SELECT COUNT(*) as count FROM users;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return row["count"] if row else 0
