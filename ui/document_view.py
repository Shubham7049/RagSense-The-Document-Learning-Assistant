"""Streamlit UI component for document upload, management, and indexing."""

from pathlib import Path

import streamlit as st

from auth.models import User
from config.settings import settings
from documents.models import DocumentStatus
from documents.repository import DocumentRepository
from ingestion.pipeline import IngestionPipeline


def format_file_size(bytes_size: int) -> str:
    """Format bytes into readable human-friendly units."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.2f} MB"


def render_document_management(user: User) -> None:
    """Render the document upload and inventory interface for the authenticated user."""
    pipeline = IngestionPipeline()
    repository = DocumentRepository()

    # Top Header & Navigation Button
    col_h1, col_h2 = st.columns([4, 1.2])
    with col_h1:
        st.markdown("### 📄 Document Management")
        st.caption("Upload and manage your documents. All data is strictly isolated to your account.")
    with col_h2:
        if st.button("💬 Go to Chat", use_container_width=True, type="primary"):
            st.session_state.current_page = "chat"
            st.rerun()

    st.write("")

    # Upload Section
    with st.expander("📤 Upload New Documents", expanded=True):
        st.markdown(
            "**Supported formats:** `PDF`, `DOCX`, `TXT`, `CSV`, `XLSX`, `JSON` &nbsp;|&nbsp; **Maximum file size:** `100 MB per file`"
        )
        uploaded_files = st.file_uploader(
            "Select files to upload and index into ChromaDB",
            type=["pdf", "docx", "txt", "csv", "xlsx", "xls", "json"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, CSV, XLSX, JSON · Maximum file size: 100 MB per file",
        )

        col_up, col_btn = st.columns([4, 1.2])
        with col_btn:
            process_btn = st.button("Index Files", use_container_width=True, type="primary", disabled=not uploaded_files)

        if process_btn and uploaded_files:
            progress_bar = st.progress(0)
            total = len(uploaded_files)
            max_bytes = settings.max_upload_size_mb * 1024 * 1024

            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.getvalue()
                if len(file_bytes) > max_bytes:
                    st.error(
                        f"❌ File '{uploaded_file.name}' ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds the maximum allowed upload size of {settings.max_upload_size_mb} MB."
                    )
                    continue

                with st.spinner(f"Ingesting '{uploaded_file.name}' with LangChain & Sentence Transformers..."):
                    record = pipeline.process_document(
                        file_name=uploaded_file.name,
                        file_bytes=file_bytes,
                        user_id=user.id,
                    )
                    if record.status == DocumentStatus.INDEXED:
                        st.success(f"Indexed **{record.filename}** ({record.chunk_count} chunks)")
                    else:
                        st.error(f"Failed to index **{record.filename}**: {record.error_message}")

                progress_bar.progress((i + 1) / total)

            st.rerun()

    # Document Inventory Section
    user_docs = repository.get_user_documents(user.id)

    if any(doc.status == DocumentStatus.INDEXED for doc in user_docs):
        col_c1, col_c2 = st.columns([3.5, 1.2])
        with col_c1:
            st.info("💡 **Ready to ask questions?** Your documents are indexed. Go to Chat to start asking questions.")
        with col_c2:
            if st.button("🚀 Start Asking Questions", use_container_width=True, type="secondary"):
                st.session_state.current_page = "chat"
                st.rerun()

    st.write("")
    col_m1, col_m2 = st.columns(2)
    total_chunks = sum(doc.chunk_count for doc in user_docs if doc.status == DocumentStatus.INDEXED)
    with col_m1:
        st.metric("Your Documents", len(user_docs))
    with col_m2:
        st.metric("Total Indexed Chunks", total_chunks)

    st.write("")
    st.markdown("#### Document Library")

    if not user_docs:
        st.info("No documents uploaded yet. Upload a PDF, DOCX, TXT, CSV, XLSX, or JSON file above to start building your knowledge base.")
        return

    # Table Header & Status Pills Styling
    st.markdown(
        """
        <style>
        .status-pill {
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .status-indexed { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
        .status-processing { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
        .status-failed { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
        .status-pending { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for doc in user_docs:
        with st.container():
            col_icon, col_name, col_meta, col_status, col_del = st.columns([0.5, 3.5, 2, 2, 1])

            # File icon by extension
            icon_map = {
                "pdf": "📕",
                "docx": "📘",
                "txt": "📄",
                "csv": "📊",
                "xlsx": "📗",
                "xls": "📗",
                "json": "🗂️",
            }
            icon = icon_map.get(doc.file_type.lower(), "📄")

            with col_icon:
                st.markdown(f"### {icon}")

            with col_name:
                st.markdown(f"**{doc.filename}**")
                st.caption(f"ID: `{doc.id[:8]}...` | {format_file_size(doc.file_size)}")

            with col_meta:
                date_str = doc.upload_timestamp.split("T")[0] if "T" in doc.upload_timestamp else doc.upload_timestamp[:10]
                st.write(f"📅 {date_str}")
                st.caption(f"Chunks: **{doc.chunk_count}**")

            with col_status:
                display_status = "READY" if doc.status == DocumentStatus.INDEXED else doc.status.value.upper()
                pill_class = f"status-{doc.status.value}"
                st.markdown(
                    f'<span class="status-pill {pill_class}">{display_status}</span>',
                    unsafe_allow_html=True,
                )
                if doc.status == DocumentStatus.FAILED and doc.error_message:
                    st.caption(f"⚠️ {doc.error_message[:40]}...")

            with col_del:
                if st.button("🗑️", key=f"del_{doc.id}", help="Delete document and vector embeddings"):
                    pipeline.delete_document(doc.id, user.id)
                    st.toast(f"Deleted {doc.filename}")
                    st.rerun()

            st.divider()
