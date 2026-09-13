"""Unit tests for Phase 2: Authentication, Password Hashing, SQLite, and User Isolation."""

import pytest

from auth.models import User, UserInDB
from auth.repository import UserRepository
from auth.service import AuthService
from auth.session import (
    get_authenticated_user_id,
    get_current_user,
    init_session,
    is_authenticated,
    login_user,
    logout_user,
)
from core.database import init_db
from core.exceptions import AuthenticationError


@pytest.fixture
def auth_service(tmp_path):
    """Fixture providing an AuthService backed by a clean temporary SQLite database."""
    test_db = tmp_path / "test_auth.db"
    init_db(test_db)
    repository = UserRepository(db_path=test_db)
    return AuthService(repository=repository)


def test_user_registration_success(auth_service):
    """Test successful user registration with valid credentials."""
    user = auth_service.register(
        username="alice",
        password="securePassword123",
        confirm_password="securePassword123",
    )
    assert user.id is not None
    assert user.id > 0
    assert user.username == "alice"
    assert user.created_at is not None

    # Verify user is in database
    user_in_db = auth_service.repository.get_by_username("alice")
    assert user_in_db is not None
    assert user_in_db.id == user.id
    assert user_in_db.username == "alice"
    assert user_in_db.password_hash is not None
    assert user_in_db.password_hash != "securePassword123"


def test_duplicate_user_registration(auth_service):
    """Test that attempting to register an existing username raises AuthenticationError."""
    auth_service.register("bob", "password123", "password123")

    with pytest.raises(AuthenticationError) as exc_info:
        auth_service.register("bob", "anotherPass456", "anotherPass456")

    assert "already taken" in str(exc_info.value)


def test_registration_validation_errors(auth_service):
    """Test input validation failures during registration."""
    # Password mismatch
    with pytest.raises(AuthenticationError) as exc_mismatch:
        auth_service.register("carol", "pass1234", "pass5678")
    assert "Passwords do not match" in str(exc_mismatch.value)

    # Password too short
    with pytest.raises(AuthenticationError) as exc_short:
        auth_service.register("dave", "123", "123")
    assert "at least 6 characters" in str(exc_short.value)

    # Username too short
    with pytest.raises(AuthenticationError) as exc_user:
        auth_service.register("ab", "password123", "password123")
    assert "at least 3 characters" in str(exc_user.value)


def test_password_hashing():
    """Test that password hashing uses unique salts and verifies correctly."""
    plain = "mySecretPassword!"
    hash1 = AuthService.hash_password(plain)
    hash2 = AuthService.hash_password(plain)

    # Hashes must never equal plaintext
    assert hash1 != plain
    # Hashes must differ due to unique salts
    assert hash1 != hash2

    # Verification must succeed with correct password
    assert AuthService.verify_password(plain, hash1) is True
    assert AuthService.verify_password(plain, hash2) is True

    # Verification must fail with incorrect password
    assert AuthService.verify_password("wrongPassword", hash1) is False
    assert AuthService.verify_password("", hash1) is False


def test_correct_login(auth_service):
    """Test successful user login with valid credentials."""
    registered = auth_service.register("eve", "strongPass123", "strongPass123")
    authenticated = auth_service.authenticate("eve", "strongPass123")

    assert authenticated.id == registered.id
    assert authenticated.username == "eve"


def test_incorrect_password(auth_service):
    """Test that login fails with incorrect password."""
    auth_service.register("frank", "correctPassword", "correctPassword")

    with pytest.raises(AuthenticationError) as exc_info:
        auth_service.authenticate("frank", "wrongPassword")

    assert "Invalid username or password" in str(exc_info.value)


def test_unknown_user_login(auth_service):
    """Test that login fails for non-existent user."""
    with pytest.raises(AuthenticationError) as exc_info:
        auth_service.authenticate("non_existent_user", "somePassword")

    assert "Invalid username or password" in str(exc_info.value)


def test_user_isolation(auth_service):
    """Test user isolation architecture: distinct users receive unique IDs."""
    user_a = auth_service.register("user_a", "passUserA123", "passUserA123")
    user_b = auth_service.register("user_b", "passUserB123", "passUserB123")

    # IDs must be strictly distinct
    assert user_a.id != user_b.id
    assert user_a.username != user_b.username

    # Repository lookup by ID verifies boundary
    fetched_a = auth_service.repository.get_by_id(user_a.id)
    fetched_b = auth_service.repository.get_by_id(user_b.id)

    assert fetched_a.id == user_a.id
    assert fetched_b.id == user_b.id
    assert fetched_a.username == "user_a"
    assert fetched_b.username == "user_b"


def test_session_management_and_isolation():
    """Test session state helpers and strict user ID isolation."""
    # Reset session
    logout_user()
    assert is_authenticated() is False
    assert get_current_user() is None

    # Enforcing isolation raises when unauthenticated
    with pytest.raises(AuthenticationError) as exc_isolation:
        get_authenticated_user_id()
    assert "No authenticated session" in str(exc_isolation.value)

    # Log in user
    mock_user = User(id=42, username="tester", created_at="2026-09-14T00:00:00Z")
    login_user(mock_user)

    assert is_authenticated() is True
    assert get_current_user() == mock_user
    assert get_authenticated_user_id() == 42

    # Logout
    logout_user()
    assert is_authenticated() is False
    assert get_current_user() is None
