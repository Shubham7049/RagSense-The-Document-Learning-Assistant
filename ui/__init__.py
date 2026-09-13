"""UI package for RAG Sense."""

from ui.auth_view import render_auth_view
from ui.chat_view import render_chat_view
from ui.document_view import render_document_management
from ui.layout import (
    apply_custom_css,
    configure_page,
    render_authenticated_dashboard,
    render_header,
    render_sidebar,
)
from ui.qa_view import render_qa_view

__all__ = [
    "apply_custom_css",
    "configure_page",
    "render_auth_view",
    "render_authenticated_dashboard",
    "render_chat_view",
    "render_document_management",
    "render_header",
    "render_qa_view",
    "render_sidebar",
]
