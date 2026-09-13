"""Authentication service with secure password hashing and business logic."""

import hashlib
import secrets
from pathlib import Path

from auth.models import User, UserLogin, UserRegistration
from auth.repository import UserRepository
from core.exceptions import AuthenticationError
from utils.logger import logger


class AuthService:
    """Provides password hashing, verification, registration, and authentication."""

    ITERATIONS = 100_000
    ALGORITHM = "sha256"

    def __init__(self, repository: UserRepository | None = None, db_path: Path | str | None = None) -> None:
        self.repository = repository or UserRepository(db_path=db_path)

    @classmethod
    def hash_password(cls, plain_password: str) -> str:
        """Hash a password using PBKDF2-HMAC-SHA256 with a unique cryptographic salt."""
        salt = secrets.token_hex(16)
        hash_bytes = hashlib.pbkdf2_hmac(
            cls.ALGORITHM,
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            cls.ITERATIONS,
        )
        return f"{salt}${hash_bytes.hex()}"

    @classmethod
    def verify_password(cls, plain_password: str, stored_hash: str) -> bool:
        """Verify a plaintext password against a stored salt$hash string using constant-time comparison."""
        try:
            salt, hash_hex = stored_hash.split("$", 1)
            computed_hash = hashlib.pbkdf2_hmac(
                cls.ALGORITHM,
                plain_password.encode("utf-8"),
                salt.encode("utf-8"),
                cls.ITERATIONS,
            ).hex()
            return secrets.compare_digest(computed_hash, hash_hex)
        except Exception:
            return False

    def register(self, username: str, password: str, confirm_password: str) -> User:
        """Validate input, hash password, and register a new user in the database."""
        # Validate via Pydantic model
        try:
            reg_data = UserRegistration(
                username=username,
                password=password,
                confirm_password=confirm_password,
            )
        except ValueError as val_err:
            raise AuthenticationError(str(val_err)) from val_err

        # Check duplicate
        existing = self.repository.get_by_username(reg_data.username)
        if existing:
            logger.warning(f"Registration attempted for existing username: '{reg_data.username}'")
            raise AuthenticationError(f"Username '{reg_data.username}' is already taken.")

        # Hash and persist
        password_hash = self.hash_password(reg_data.password)
        user = self.repository.create_user(
            username=reg_data.username,
            password_hash=password_hash,
        )
        logger.info(f"User '{user.username}' successfully registered with ID #{user.id}.")
        return user

    def authenticate(self, username: str, password: str) -> User:
        """Authenticate user credentials and return public User model."""
        try:
            login_data = UserLogin(username=username, password=password)
        except ValueError as val_err:
            raise AuthenticationError(str(val_err)) from val_err

        user_in_db = self.repository.get_by_username(login_data.username)
        if not user_in_db:
            logger.warning(f"Authentication failed: user '{login_data.username}' not found.")
            raise AuthenticationError("Invalid username or password.")

        if not self.verify_password(login_data.password, user_in_db.password_hash):
            logger.warning(f"Authentication failed: incorrect password for user '{login_data.username}'.")
            raise AuthenticationError("Invalid username or password.")

        logger.info(f"User '{user_in_db.username}' (ID #{user_in_db.id}) logged in successfully.")
        return User(
            id=user_in_db.id,
            username=user_in_db.username,
            created_at=user_in_db.created_at,
        )
