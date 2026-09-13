"""Polished ChatGPT-like conversational UI view with suggestions, streaming, and sources."""

import streamlit as st

from chat.models import ChatThread
from chat.service import ChatService
from config.settings import settings
from core.exceptions import LLMError
from documents.models import DocumentStatus
from documents.repository import DocumentRepository
from rag.service import RAGService
from utils.logger import logger


def format_citations_block(citations: list[dict]) -> str:
    """Format structured citations into an HTML collapsible details block for persistent rendering."""
    if not citations:
        return ""

    items_html = []
    for cite in citations:
        page_info = f" — Page {cite['page_number']}" if cite.get("page_number") else ""
        chunk_info = f" · Chunk #{cite['chunk_index']}" if cite.get("chunk_index") is not None else ""
        items_html.append(
            f"<div style='margin-bottom: 0.5rem; font-size: 0.88rem;'>"
            f"📄 <b>{cite['filename']}</b>{page_info}{chunk_info}<br>"
            f"<span style='color: #64748b; font-size: 0.82rem; margin-left: 1.2rem; font-style: italic;'>\"{cite['snippet']}\"</span>"
            f"</div>"
        )

    all_items = "".join(items_html)
    return (
        f"\n\n<details style='margin-top: 1rem; padding: 0.5rem 0.8rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;'>"
        f"<summary style='cursor: pointer; font-weight: 600; color: #4338ca;'>📚 Sources ({len(citations)})</summary>"
        f"<div style='margin-top: 0.6rem;'>{all_items}</div>"
        f"</details>"
    )


def render_chat_view(user_id: int) -> None:
    """Render the polished ChatGPT-like conversational view."""
    chat_service = ChatService()
    rag_service = RAGService()
    doc_repo = DocumentRepository()

    # 1. Resolve active chat thread
    active_chat_id = st.session_state.get("active_chat_id")
    user_chats = chat_service.list_user_chats(user_id=user_id)

    if not active_chat_id or not any(c.id == active_chat_id for c in user_chats):
        if user_chats:
            active_chat_id = user_chats[0].id
        else:
            new_chat = chat_service.create_new_chat(user_id=user_id, title="New Chat")
            active_chat_id = new_chat.id
            user_chats = [new_chat]
        st.session_state.active_chat_id = active_chat_id

    current_chat: ChatThread | None = chat_service.get_chat(active_chat_id, user_id)
    if not current_chat:
        st.error("Selected conversation could not be loaded.")
        return

    # 2. Check Groq API Key configuration
    from config.settings import get_settings, reload_settings
    current_settings = get_settings()
    if not current_settings.groq_api_key or current_settings.groq_api_key.strip() in {"", "your_groq_api_key_here"}:
        current_settings = reload_settings()

    if not current_settings.groq_api_key or current_settings.groq_api_key.strip() in {"", "your_groq_api_key_here"}:
        st.warning(
            "⚠️ **Groq API Key Not Configured**: Please configure `GROQ_API_KEY` in your `.env` file to enable AI answers."
        )

    # 3. Document Scope & Retrieval Controls Header
    user_docs = doc_repo.list_documents(user_id=user_id)
    indexed_docs = [d for d in user_docs if d.status == DocumentStatus.INDEXED]

    col_title, col_scope, col_k = st.columns([5, 4, 2])
    with col_title:
        st.markdown(f"### 💬 {current_chat.title}")

    with col_scope:
        doc_options = {"all": "🌐 All My Documents"}
        for d in indexed_docs:
            doc_options[d.id] = f"📄 {d.filename}"

        selected_scope = st.selectbox(
            "Document Scope",
            options=list(doc_options.keys()),
            format_func=lambda k: doc_options[k],
            index=0,
            key=f"scope_{active_chat_id}",
            help="Restrict retrieval to a single document or search all indexed documents.",
        )
        selected_doc_id = None if selected_scope == "all" else selected_scope

    with col_k:
        top_k = st.number_input(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=settings.retrieval_k,
            step=1,
            key=f"k_{active_chat_id}",
            help="Number of document chunks to retrieve.",
        )

    st.divider()

    # 4. Render Message Timeline or Empty State
    messages = chat_service.get_chat_messages(chat_id=active_chat_id, user_id=user_id)
    prompt_to_run = None

    if not messages:
        # Polished Empty State
        st.markdown(
            """
            <div style="text-align: center; padding: 2rem 1rem 1.2rem 1rem;">
                <h1 style="font-size: 2.3rem; font-weight: 800; margin-bottom: 0.3rem;">🧠 RAG Sense</h1>
                <p style="font-size: 1.15rem; color: #64748b; margin-bottom: 1.5rem;">
                    Understand your documents. Ask anything.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 💡 Suggested Questions")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("📝 Summarize my document", use_container_width=True, key="sug_sum"):
                prompt_to_run = "Summarize my document and outline the main points."
            if st.button("🔑 What are the key points?", use_container_width=True, key="sug_pts"):
                prompt_to_run = "What are the key points and essential takeaways?"
            if st.button("💡 Explain the main concepts", use_container_width=True, key="sug_exp"):
                prompt_to_run = "Explain the main concepts and core ideas in detail."
        with col_s2:
            if st.button("🎯 What skills are mentioned?", use_container_width=True, key="sug_skills"):
                prompt_to_run = "What skills, qualifications, or key capabilities are mentioned?"
            if st.button("❓ Create questions from this document", use_container_width=True, key="sug_q"):
                prompt_to_run = "Create study and comprehension questions based on this document."

        if not indexed_docs:
            st.write("")
            col_inf, col_up_btn = st.columns([3.5, 1.3])
            with col_inf:
                st.info("ℹ️ **No documents indexed yet**: Upload documents to ground the AI in your own files.")
            with col_up_btn:
                if st.button("📤 Upload Documents", use_container_width=True, type="primary"):
                    st.session_state.current_page = "documents"
                    st.rerun()
    else:
        for msg in messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content, unsafe_allow_html=True)

    # 5. Question Input & Execution
    chat_input_val = st.chat_input(
        "Ask anything about your documents...",
        key=f"chat_input_{active_chat_id}",
    )

    query = prompt_to_run or chat_input_val

    if query and query.strip():
        clean_prompt = query.strip()

        # Auto-update initial "New Chat" title
        if current_chat.title == "New Chat" and len(messages) == 0:
            new_title = ChatService.generate_title_from_prompt(clean_prompt)
            chat_service.rename_chat(active_chat_id, user_id, new_title)
            current_chat.title = new_title

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(clean_prompt)

        # Save user message to persistent SQLite
        chat_service.add_message(
            chat_id=active_chat_id,
            user_id=user_id,
            role="user",
            content=clean_prompt,
        )

        # Retrieve contextual history (Recent + Relevant Older)
        recent_hist, older_hist = chat_service.get_contextual_history(
            chat_id=active_chat_id,
            user_id=user_id,
            query=clean_prompt,
        )

        # Stream assistant response
        with st.chat_message("assistant"):
            try:
                stream_gen, citations, docs = rag_service.ask_stream(
                    question=clean_prompt,
                    user_id=user_id,
                    document_id=selected_doc_id,
                    k=top_k,
                    recent_history=recent_hist,
                    relevant_history=older_hist,
                )
                raw_answer = st.write_stream(stream_gen)

                full_answer = str(raw_answer) if raw_answer else ""

                # Format and render sources
                citations_html = format_citations_block(citations)
                if citations_html:
                    st.markdown(citations_html, unsafe_allow_html=True)

                # Persist assistant response with formatted sources
                content_to_save = f"{full_answer}{citations_html}" if full_answer else ""
                if content_to_save:
                    chat_service.add_message(
                        chat_id=active_chat_id,
                        user_id=user_id,
                        role="assistant",
                        content=content_to_save,
                    )

            except LLMError as e:
                st.error(f"❌ {e.message}")
            except Exception as e:
                logger.error(f"Unexpected RAG error: {e}", exc_info=True)
                st.error("❌ An error occurred during response generation. Please check your settings or try again.")

        st.rerun()
