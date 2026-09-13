"""UI layout and navigation for RAG Sense."""

from datetime import datetime, timedelta, timezone

import streamlit as st

from auth.models import User
from auth.session import logout_user
from chat.models import ChatThread
from chat.service import ChatService
from config.settings import settings
from documents.models import DocumentStatus
from documents.repository import DocumentRepository
from ui.chat_view import render_chat_view
from ui.document_view import render_document_management


def configure_page() -> None:
    """Configure primary Streamlit page settings."""
    st.set_page_config(
        page_title=f"{settings.app_name} - Understand your documents. Ask anything.",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css() -> None:
    """Inject subtle, modern styling for a clean ChatGPT-like visual hierarchy."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #4f46e5, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.2rem;
        }
        .isolation-badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            font-size: 0.82rem;
            font-weight: 600;
            border-radius: 9999px;
            background-color: #eff6ff;
            color: #2563eb;
            border: 1px solid #bfdbfe;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(user: User | None = None) -> None:
    """Render top application header, showing user identity if authenticated."""
    col1, col2 = st.columns([4, 1.5])
    with col1:
        st.markdown(f'<div class="main-header">{settings.app_name}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Understand your documents. Ask anything.</div>',
            unsafe_allow_html=True,
        )
    with col2:
        if user:
            st.markdown(
                f"""
                <div style="text-align: right; padding-top: 0.5rem;">
                    <span class="isolation-badge">👤 {user.username}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def group_chats_by_date(chats: list[ChatThread]) -> dict[str, list[ChatThread]]:
    """Group chat threads into Today, Yesterday, and Older."""
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    groups: dict[str, list[ChatThread]] = {
        "Today": [],
        "Yesterday": [],
        "Older": [],
    }
    for chat in chats:
        try:
            chat_date = datetime.fromisoformat(chat.updated_at).date()
        except Exception:
            chat_date = today

        if chat_date == today:
            groups["Today"].append(chat)
        elif chat_date == yesterday:
            groups["Yesterday"].append(chat)
        else:
            groups["Older"].append(chat)

    return {k: v for k, v in groups.items() if v}


def render_sidebar(user: User | None = None) -> None:
    """Render persistent sidebar with user identity, chat threads, documents, and controls."""
    with st.sidebar:
        st.markdown(f"### 🧠 {settings.app_name}")
        st.caption("Understand your documents. Ask anything.")

        if user:
            st.divider()
            # 1. New Chat Button
            if st.button("➕ New Chat", use_container_width=True, type="primary", key="btn_new_chat"):
                chat_service = ChatService()
                new_chat = chat_service.create_new_chat(user_id=user.id, title="New Chat")
                st.session_state.active_chat_id = new_chat.id
                st.session_state.current_page = "chat"
                st.rerun()

            # 2. Conversation Threads
            st.markdown("### 💬 Chats")
            chat_service = ChatService()
            user_chats = chat_service.list_user_chats(user_id=user.id)

            if not user_chats:
                st.caption("No conversations yet. Click **➕ New Chat** to start.")
            else:
                active_id = st.session_state.get("active_chat_id")
                if not active_id or not any(c.id == active_id for c in user_chats):
                    active_id = user_chats[0].id
                    st.session_state.active_chat_id = active_id

                grouped = group_chats_by_date(user_chats)
                for period, period_chats in grouped.items():
                    st.caption(f"**{period}**")
                    for c in period_chats:
                        is_active = (c.id == active_id)
                        btn_label = f"👉 {c.title}" if is_active else f"💬 {c.title}"
                        if st.button(
                            btn_label,
                            key=f"sidebar_chat_{c.id}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary",
                        ):
                            st.session_state.active_chat_id = c.id
                            st.session_state.current_page = "chat"
                            st.rerun()

                        # Compact management for the active thread
                        if is_active:
                            with st.expander("⚙️ Manage Thread", expanded=False):
                                new_name = st.text_input("Rename", value=c.title, key=f"rename_{c.id}")
                                col_r, col_d = st.columns(2)
                                with col_r:
                                    if st.button("Save", key=f"save_r_{c.id}", use_container_width=True):
                                        if new_name.strip() and new_name.strip() != c.title:
                                            chat_service.rename_chat(c.id, user.id, new_name.strip())
                                            st.rerun()
                                with col_d:
                                    if st.button(
                                        "Delete",
                                        key=f"del_{c.id}",
                                        use_container_width=True,
                                        type="secondary",
                                    ):
                                        chat_service.delete_chat(c.id, user.id)
                                        remaining = [ch for ch in user_chats if ch.id != c.id]
                                        st.session_state.active_chat_id = remaining[0].id if remaining else None
                                        st.rerun()

            # 3. Documents Navigation & Summary
            st.divider()
            st.markdown("### 📑 Documents")
            if st.button("📚 Manage Documents", use_container_width=True, key="btn_nav_documents"):
                st.session_state.current_page = "documents"
                st.rerun()

            doc_repo = DocumentRepository()
            user_docs = doc_repo.list_documents(user_id=user.id)
            if not user_docs:
                st.caption("No documents indexed yet.")
            else:
                for d in user_docs[:4]:
                    status_pill = (
                        "🟢 Ready"
                        if d.status == DocumentStatus.INDEXED
                        else ("🟡 Processing" if d.status == DocumentStatus.PROCESSING else "🔴 Failed")
                    )
                    short_name = d.filename if len(d.filename) <= 16 else f"{d.filename[:14]}.."
                    st.caption(f"• **{short_name}** `{d.file_type}` {status_pill}")
                if len(user_docs) > 4:
                    st.caption(f"*+ {len(user_docs) - 4} more document(s)*")

            # 4. Account info & Logout
            st.divider()
            st.markdown("### 👤 Account")
            st.markdown(f"**User:** `{user.username}`")
            st.write("")
            if st.button("🚪 Log Out", use_container_width=True, type="secondary"):
                logout_user()
                st.rerun()


def render_authenticated_dashboard(user: User) -> None:
    """Render the main dashboard based on current active view: Chat or Documents."""
    current_page = st.session_state.get("current_page", "chat")

    if current_page == "documents":
        render_document_management(user)
    else:
        render_chat_view(user.id)
