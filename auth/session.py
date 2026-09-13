"""Session management and user isolation helpers for Streamlit."""

import streamlit as st

from auth.models import User
from core.exceptions import AuthenticationError

SESSION_USER_KEY = "authenticated_user"


def init_session() -> None:
    """Initialize authentication session keys in Streamlit session state."""
    if SESSION_USER_KEY not in st.session_state:
        st.session_state[SESSION_USER_KEY] = None


def get_current_user() -> User | None:
    """Return the currently authenticated user, or None if not logged in."""
    init_session()
    return st.session_state.get(SESSION_USER_KEY)


def is_authenticated() -> bool:
    """Check if there is an active authenticated user session."""
    return get_current_user() is not None


def login_user(user: User) -> None:
    """Store the authenticated user in session state."""
    init_session()
    st.session_state[SESSION_USER_KEY] = user


def logout_user() -> None:
    """Clear user session state and return user to login screen."""
    init_session()
    st.session_state[SESSION_USER_KEY] = None


def get_authenticated_user_id() -> int:
    """Enforce user isolation by returning the active user ID.

    Raises AuthenticationError if no user is currently authenticated.
    Every future document, chat, and vector search operation MUST call this
    function to bind data access exclusively to the active user.
    """
    user = get_current_user()
    if not user:
        raise AuthenticationError("No authenticated session active. Operation requires an active user ID.")
    return user.id
