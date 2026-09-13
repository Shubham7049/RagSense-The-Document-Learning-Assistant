"""Main entrypoint for RAG Sense Streamlit application."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from auth.session import get_current_user
from config.settings import settings
from core.database import init_db
from core.exceptions import RAGSenseError
from ui.auth_view import render_auth_view
from ui.layout import (
    apply_custom_css,
    configure_page,
    render_authenticated_dashboard,
    render_header,
    render_sidebar,
)
from utils.helpers import init_project_directories
from utils.logger import logger


def main() -> None:
    """Bootstrap and run the RAG Sense application."""
    try:
        # Initialize directories (data, logs)
        init_project_directories()

        # Initialize SQLite database schema
        init_db()

        # Configure Streamlit page and styling
        configure_page()
        apply_custom_css()

        # Determine authentication state
        current_user = get_current_user()

        # Render sidebar with active user context
        render_sidebar(user=current_user)

        # Render main content based on authentication state
        if current_user:
            render_header(user=current_user)
            render_authenticated_dashboard(user=current_user)
        else:
            render_auth_view()

    except RAGSenseError as rse:
        logger.error(f"RAG Sense Error: {rse}", exc_info=settings.debug)
        st.error(f"**Application Error**: {rse.message}")
        if settings.debug and rse.details:
            st.json(rse.details)
    except Exception as exc:
        logger.critical(f"Unhandled runtime error: {exc}", exc_info=True)
        st.error(f"**Unexpected Error**: {exc}")
        if settings.debug:
            st.exception(exc)


if __name__ == "__main__":
    main()
