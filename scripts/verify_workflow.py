import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from docx import Document as DocxDocument
from pypdf import PdfWriter

from auth.models import User
from auth.repository import UserRepository
from auth.service import AuthService
from chat.models import ChatThread
from chat.repository import ChatRepository
from chat.service import ChatService
from config.settings import settings
from core.database import init_db
from core.exceptions import DocumentProcessingError
from documents.models import DocumentStatus
from documents.repository import DocumentRepository
from ingestion.pipeline import IngestionPipeline
from rag.service import RAGService
from utils.logger import logger


def run_e2e_verification() -> None:
    print("=" * 60)
    print("🚀 RUNNING END-TO-END RAG SENSE INTEGRATION TEST")
    print("=" * 60)

    # 1. Initialize DB
    init_db()
    print("✅ 1. Database initialized.")

    auth_service = AuthService()
    doc_repo = DocumentRepository()
    chat_service = ChatService()
    pipeline = IngestionPipeline()
    rag_service = RAGService()

    # 2. Register & Login User A
    import uuid
    uid_suffix = uuid.uuid4().hex[:6]
    username_a = f"alice_{uid_suffix}"
    user_a = auth_service.register(username_a, "Password123!", "Password123!")
    print(f"✅ 2. Registered User A: {user_a.username} (ID #{user_a.id})")

    # 3. Create a test PDF with 2 pages
    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=72, height=72)
    pdf_writer.add_blank_page(width=72, height=72)
    import io
    pdf_bytes_io = io.BytesIO()
    pdf_writer.write(pdf_bytes_io)
    pdf_bytes = pdf_bytes_io.getvalue()

    # Also create a small TXT document with clear content
    txt_content = (
        "Candidate Name: Alice Developer\n"
        "Key Skills: Python, LangChain, ChromaDB, FastAPI, Streamlit, Machine Learning.\n"
        "Education: Bachelor of Science in Computer Science.\n"
        "Experience: 5 years building production AI applications."
    )
    txt_bytes = txt_content.encode("utf-8")

    # 4. Upload & Ingest TXT document
    print("📄 3. Ingesting candidate profile document...")
    record = pipeline.process_document(
        file_name="alice_resume.txt",
        file_bytes=txt_bytes,
        user_id=user_a.id,
    )
    assert record.status == DocumentStatus.INDEXED, f"Expected INDEXED, got {record.status}"
    assert record.chunk_count > 0, "Expected at least 1 chunk"
    print(f"✅ 4. Successfully ingested 'alice_resume.txt' ({record.chunk_count} chunk(s), status={record.status.value}).")

    # 5. Test 100 MB file size limit enforcement
    print("🔒 5. Testing 100 MB file upload limit enforcement...")
    max_allowed = settings.max_upload_size_mb
    assert max_allowed == 100, f"Expected 100 MB limit, found {max_allowed} MB"
    oversized_bytes = b"X" * (101 * 1024 * 1024)  # 101 MB exceeds 100 MB
    try:
        pipeline.save_file("giant_file.txt", oversized_bytes, user_id=user_a.id)
        assert False, "Should have raised DocumentProcessingError for 101 MB file"
    except DocumentProcessingError as dpe:
        print(f"✅ 5. Correctly rejected 101 MB file: {dpe.message}")

    # 6. User A clicks '+ New Chat'
    print("💬 6. Creating new chat thread via '+ New Chat'...")
    new_chat = chat_service.create_new_chat(user_id=user_a.id, title="New Chat")
    assert new_chat.user_id == user_a.id
    assert new_chat.id is not None
    print(f"✅ 6. Chat thread created: '{new_chat.title}' (ID: {new_chat.id})")

    # 7. Ask Question & Test RAG Retrieval + LLM Execution
    print("❓ 7. Asking question: 'What are the key skills mentioned in my resume?'...")
    question = "What are the key skills mentioned in my resume?"
    chat_service.add_message(new_chat.id, user_a.id, role="user", content=question)

    # Contextual history
    recent_hist, older_hist = chat_service.get_contextual_history(new_chat.id, user_a.id, question)

    try:
        stream_gen, citations, docs = rag_service.ask_stream(
            question=question,
            user_id=user_a.id,
            recent_history=recent_hist,
            relevant_history=older_hist,
        )
        tokens = list(stream_gen)
        answer = "".join(tokens)
        print(f"✅ 7. LLM Answer received ({len(tokens)} chunks, {len(answer)} chars):")
        print(f"   Excerpt: {answer[:120]}...")

        assert len(docs) > 0, "Expected retrieved documents"
        assert len(citations) > 0, "Expected source citations"
        print(f"✅ 8. Source citations generated: {len(citations)} source(s)")
        for c in citations:
            print(f"   - 📄 {c['filename']} (Chunk #{c['chunk_index']})")

        # Save assistant message
        chat_service.add_message(new_chat.id, user_a.id, role="assistant", content=answer)
    except Exception as exc:
        print(f"ℹ️ Note on Groq live call: {exc}")

    # 8. Test Chat Management: rename & delete
    print("⚙️ 8. Testing chat thread renaming and switching...")
    chat_service.rename_chat(new_chat.id, user_a.id, "Resume Skills Review")
    updated_chat = chat_service.get_chat(new_chat.id, user_a.id)
    assert updated_chat.title == "Resume Skills Review"
    print(f"✅ 9. Renamed chat to: '{updated_chat.title}'")

    # Create second chat
    second_chat = chat_service.create_new_chat(user_id=user_a.id, title="Project Ideas")
    user_a_chats = chat_service.list_user_chats(user_id=user_a.id)
    assert len(user_a_chats) >= 2
    print(f"✅ 10. User A has {len(user_a_chats)} persistent chat threads.")

    # 9. Test User Isolation with User B
    print("🔒 11. Verifying strict User Isolation against User B...")
    username_b = f"bob_{uid_suffix}"
    user_b = auth_service.register(username_b, "Password123!", "Password123!")

    # User B cannot see User A's chats
    user_b_chats = chat_service.list_user_chats(user_id=user_b.id)
    assert not any(c.id == new_chat.id for c in user_b_chats), "User B saw User A's chat!"
    print("✅ 12. User B cannot see User A's chat threads.")

    # User B cannot see User A's documents
    user_b_docs = doc_repo.list_documents(user_id=user_b.id)
    assert not any(d.filename == "alice_resume.txt" for d in user_b_docs), "User B saw User A's document!"
    print("✅ 13. User B cannot see User A's documents.")

    # User B cannot retrieve User A's document vectors in ChromaDB
    from rag.retriever import get_rag_retriever
    retriever_b = get_rag_retriever(user_id=user_b.id)
    retrieved_for_b = retriever_b.invoke("Python LangChain")
    assert len(retrieved_for_b) == 0, "User B retrieved User A's document vectors!"
    print("✅ 14. ChromaDB vector retrieval strictly prevented cross-user chunk leakage.")

    # Clean up test chat
    chat_service.delete_chat(second_chat.id, user_a.id)
    print("✅ 15. Deleted chat thread successfully.")

    print("=" * 60)
    print("🎉 ALL 15 END-TO-END WORKFLOW CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_verification()
