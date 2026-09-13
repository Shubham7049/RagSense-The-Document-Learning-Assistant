"""Authentication package for RAG Sense."""

from auth.models import User, UserInDB, UserLogin, UserRegistration
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

__all__ = [
    "AuthService",
    "User",
    "UserInDB",
    "UserLogin",
    "UserRegistration",
    "UserRepository",
    "get_authenticated_user_id",
    "get_current_user",
    "init_session",
    "is_authenticated",
    "login_user",
    "logout_user",
]
