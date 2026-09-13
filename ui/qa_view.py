"""Interactive Streamlit UI for RAG Question Answering with ChatGroq & Streaming."""

import streamlit as st

from auth.models import User
from config.settings import settings
from core.exceptions import LLMError
from documents.models import DocumentStatus
from documents.repository import DocumentRepository
from rag.service import RAGService


def render_qa_view(user: User) -> None:
    """Render the document question-answering interface with streaming and citations."""
    repository = DocumentRepository()
    rag_service = RAGService()

    st.markdown("### 💬 Document Q&A Assistant")
    st.caption("Ask questions grounded strictly in your uploaded knowledge base. Responses stream in real-time.")

    # Check Groq API configuration
    has_api_key = bool(
        settings.groq_api_key
        and settings.groq_api_key.strip()
        and settings.groq_api_key.strip() != "your_groq_api_key_here"
    )

    if not has_api_key:
        st.warning(
            "⚠️ **Groq API Key Required**\n\n"
            "To enable AI question-answering with **ChatGroq**, please configure your API key in the `.env` file:\n\n"
            "```env\nGROQ_API_KEY=gsk_your_actual_key_here\n```\n\n"
            "*You can obtain a free, high-speed API key at [console.groq.com](https://console.groq.com/keys).*"
        )

    # Fetch user's indexed documents
    user_docs = repository.get_user_documents(user.id)
    indexed_docs = [d for d in user_docs if d.status == DocumentStatus.INDEXED]

    if not indexed_docs:
        st.info(
            "📁 **No Indexed Documents Available**\n\n"
            "You have not indexed any documents yet. Please switch to the **Document Management** tab "
            "to upload and index your files first."
        )
        return

    # Retrieval Controls
    with st.expander("🎯 Retrieval Settings & Filtering", expanded=False):
        col_f1, col_f2 = st.columns([3, 1])

        with col_f1:
            doc_options = {"all": "🌐 All Indexed Documents"}
            for d in indexed_docs:
                doc_options[d.id] = f"📄 {d.filename} ({d.chunk_count} chunks)"

            selected_doc_key = st.selectbox(
                "Scope Query To:",
                options=list(doc_options.keys()),
                format_func=lambda k: doc_options[k],
                key="rag_doc_filter",
            )
            selected_doc_id = None if selected_doc_key == "all" else selected_doc_key

        with col_f2:
            top_k = st.slider("Top Chunks (k)", min_value=1, max_value=10, value=settings.retrieval_k, key="rag_top_k")

    st.write("")

    # Question Form
    with st.form(key="qa_form", clear_on_submit=False):
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. Explain the main topic covered in my documents...",
            key="qa_question_input",
        )
        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            submit_btn = st.form_submit_button("Ask Question", use_container_width=True, type="primary")
        with col_info:
            st.caption(f"Model: `{settings.groq_model}` | Context strictly isolated to User #{user.id}")

    # Process and Stream Answer
    if submit_btn:
        if not question or not question.strip():
            st.warning("Please enter a question.")
            return

        if not has_api_key:
            st.error("Cannot query model: GROQ_API_KEY is not configured.")
            return

        st.markdown(f"#### ❓ Question: *{question.strip()}*")

        try:
            with st.spinner("Retrieving relevant document chunks..."):
                stream_gen, citations, docs = rag_service.ask_stream(
                    question=question,
                    user_id=user.id,
                    document_id=selected_doc_id,
                    k=top_k,
                )

            st.markdown("#### 💡 Answer:")
            # Progressive token streaming via Streamlit's native write_stream
            st.write_stream(stream_gen)

            # Source Citations Display
            st.write("")
            with st.expander(f"📚 Source Citations & References ({len(citations)} chunk(s) retrieved)", expanded=False):
                if not citations:
                    st.write("No matching document chunks were returned for this query.")
                else:
                    for c in citations:
                        st.markdown(f"**Reference {c['index']}: `{c['filename']}`** (Chunk #{c['chunk_index']})")
                        st.caption(f"Document ID: `{c['document_id']}`")
                        st.markdown(f"> *\"{c['snippet']}\"*")
                        st.divider()

        except LLMError as llm_err:
            st.error(f"**LLM Service Error**: {llm_err.message}")
        except Exception as err:
            st.error(f"**An error occurred**: {err}")
