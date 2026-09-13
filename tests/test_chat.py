"""Unit tests for Phase 5: Multi-Chat Threads, Persistence, History Retrieval, and User Isolation."""

import pytest
from langchain_community.chat_models import FakeListChatModel

from chat.history import format_messages_for_prompt
from chat.models import ChatMessage, ChatThread
from chat.repository import ChatRepository
from chat.service import ChatService
from core.database import init_db
from core.exceptions import RAGSenseError
from rag.service import RAGService


@pytest.fixture
def test_db_path(tmp_path):
    """Provide a temporary initialized SQLite database path with seeded users."""
    db_file = tmp_path / "test_chat.db"
    init_db(db_file)

    # Seed users to satisfy foreign key constraints
    from core.database import get_db_context
    with get_db_context(db_file) as conn:
        for uid in range(1, 100):
            conn.execute(
                "INSERT OR IGNORE INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?);",
                (uid, f"user_{uid}", "dummy_hash", "2026-09-14T00:00:00Z"),
            )
    return db_file


@pytest.fixture
def chat_service(test_db_path, tmp_path):
    """Provide an isolated ChatService instance with a temporary ChromaDB directory."""
    chroma_dir = tmp_path / "test_chat_chroma"
    return ChatService(
        db_path=test_db_path,
        persist_dir=chroma_dir,
        collection_name="test_chat_history",
    )


def test_create_and_list_chats(chat_service):
    """Test chat thread creation and listing ordered by updated_at."""
    # User 1 creates two chats
    chat1 = chat_service.create_new_chat(user_id=1, title="Physics Discussion")
    chat2 = chat_service.create_new_chat(user_id=1, title="Calculus Notes")

    assert chat1.id != chat2.id
    assert chat1.user_id == 1
    assert chat2.title == "Calculus Notes"

    # Listing chats
    chats = chat_service.list_user_chats(user_id=1)
    assert len(chats) == 2
    # chat2 was created last so it should appear first
    assert chats[0].id == chat2.id
    assert chats[1].id == chat1.id


def test_rename_and_delete_chat(chat_service):
    """Test renaming a chat thread and deleting it."""
    chat = chat_service.create_new_chat(user_id=10, title="Initial Title")

    # Rename
    success = chat_service.rename_chat(chat.id, user_id=10, new_title="Updated Title")
    assert success is True

    fetched = chat_service.get_chat(chat.id, user_id=10)
    assert fetched is not None
    assert fetched.title == "Updated Title"

    # Empty rename fails
    with pytest.raises(ValueError):
        chat_service.rename_chat(chat.id, user_id=10, new_title="   ")

    # Delete
    deleted = chat_service.delete_chat(chat.id, user_id=10)
    assert deleted is True

    # Confirm deletion
    assert chat_service.get_chat(chat.id, user_id=10) is None
    assert len(chat_service.list_user_chats(user_id=10)) == 0


def test_user_chat_isolation(chat_service):
    """Verify that User A cannot see, read, rename, or delete User B's chats."""
    # User 1 creates a chat
    chat_u1 = chat_service.create_new_chat(user_id=1, title="User 1 Confidential")

    # User 2 creates a chat
    chat_u2 = chat_service.create_new_chat(user_id=2, title="User 2 Project")

    # User 1 lists chats
    u1_chats = chat_service.list_user_chats(user_id=1)
    assert len(u1_chats) == 1
    assert u1_chats[0].id == chat_u1.id

    # User 2 lists chats
    u2_chats = chat_service.list_user_chats(user_id=2)
    assert len(u2_chats) == 1
    assert u2_chats[0].id == chat_u2.id

    # User 2 cannot access User 1's chat
    assert chat_service.get_chat(chat_u1.id, user_id=2) is None

    # User 2 cannot rename User 1's chat
    assert chat_service.rename_chat(chat_u1.id, user_id=2, new_title="Hacked Title") is False
    assert chat_service.get_chat(chat_u1.id, user_id=1).title == "User 1 Confidential"

    # User 2 cannot delete User 1's chat
    assert chat_service.delete_chat(chat_u1.id, user_id=2) is False
    assert chat_service.get_chat(chat_u1.id, user_id=1) is not None


def test_message_persistence_and_loading(chat_service):
    """Test adding messages and loading them in chronological order."""
    chat = chat_service.create_new_chat(user_id=5, title="Q&A Session")

    # Add messages
    m1 = chat_service.add_message(chat.id, user_id=5, role="user", content="What is photosynthesis?")
    m2 = chat_service.add_message(chat.id, user_id=5, role="assistant", content="Photosynthesis is the process...")
    m3 = chat_service.add_message(chat.id, user_id=5, role="user", content="Where does it take place?")

    messages = chat_service.get_chat_messages(chat.id, user_id=5)
    assert len(messages) == 3
    assert messages[0].content == "What is photosynthesis?"
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[2].content == "Where does it take place?"

    # User 6 cannot read User 5's messages
    u6_messages = chat_service.get_chat_messages(chat.id, user_id=6)
    assert len(u6_messages) == 0


def test_recent_history_selection(chat_service):
    """Test that recent history selects only the last N messages chronologically."""
    chat = chat_service.create_new_chat(user_id=7, title="Long Thread")

    # Insert 10 messages
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        chat_service.add_message(chat.id, user_id=7, role=role, content=f"Message #{i}")

    # Request recent 4
    recent_str, older_str = chat_service.get_contextual_history(
        chat_id=chat.id,
        user_id=7,
        query="test query",
        recent_k=4,
        older_k=2,
    )

    # Should contain messages 6, 7, 8, 9
    assert "Message #6" in recent_str
    assert "Message #7" in recent_str
    assert "Message #8" in recent_str
    assert "Message #9" in recent_str
    # Should NOT contain messages 0, 1, 2, 3 in recent
    assert "Message #0" not in recent_str
    assert "Message #1" not in recent_str


def test_relevant_older_history_retrieval(chat_service):
    """Test that semantically relevant older messages are retrieved while excluding recent ones."""
    chat = chat_service.create_new_chat(user_id=9, title="Quantum Computing")

    # Older messages (about superconductors and qubits)
    chat_service.add_message(
        chat.id,
        user_id=9,
        role="user",
        content="Superconductors operate at extremely low cryogenic temperatures.",
    )
    chat_service.add_message(
        chat.id,
        user_id=9,
        role="assistant",
        content="Cryogenic cooling preserves quantum coherence in superconducting qubits.",
    )

    # Filler intermediate messages
    for idx in range(6):
        chat_service.add_message(chat.id, user_id=9, role="user", content=f"Filler conversation topic {idx}")

    # Recent messages (about something else)
    chat_service.add_message(chat.id, user_id=9, role="user", content="What about classical binary bits?")
    chat_service.add_message(chat.id, user_id=9, role="assistant", content="Classical bits are either 0 or 1.")

    # Query specifically about cryogenics / superconductors
    recent_str, older_str = chat_service.get_contextual_history(
        chat_id=chat.id,
        user_id=9,
        query="What temperature is needed for superconductors?",
        recent_k=4,
        older_k=2,
    )

    # Recent history should contain the classical bit discussion
    assert "classical binary bits" in recent_str or "Classical bits are either 0 or 1" in recent_str
    # Older relevant history should have retrieved the superconductor/cryogenic discussion
    assert "Superconductors" in older_str or "Cryogenic" in older_str


def test_title_generation_helper():
    """Verify auto-title generation from user queries without LLM overhead."""
    t1 = ChatService.generate_title_from_prompt("What is the difference between supervised and unsupervised learning?")
    assert len(t1) <= 38
    assert "difference between" in t1.lower()

    t2 = ChatService.generate_title_from_prompt("### Python 3.12 features: Explain PEP 695!")
    assert "Python 3.12 features" in t2

    t3 = ChatService.generate_title_from_prompt("   ")
    assert t3 == "New Chat"


def test_format_messages_for_prompt():
    """Test formatting ChatMessages into a dialog block."""
    assert format_messages_for_prompt([]) == "None"

    msgs = [
        ChatMessage(id="1", chat_id="c1", user_id=1, role="user", content="Hello", created_at="2026-09-14T00:00:00"),
        ChatMessage(id="2", chat_id="c1", user_id=1, role="assistant", content="Hi there!", created_at="2026-09-14T00:00:01"),
    ]
    formatted = format_messages_for_prompt(msgs)
    assert "User: Hello" in formatted
    assert "Assistant: Hi there!" in formatted


def test_rag_with_conversation_history(tmp_path):
    """Test full RAG pipeline integration with document context and conversation history."""
    chroma_dir = tmp_path / "rag_history_chroma"
    fake_llm = FakeListChatModel(responses=["Answer incorporating history and document."])

    service = RAGService(
        groq_api_key="gsk_dummy_key",
        persist_dir=chroma_dir,
        collection_name="test_rag_hist_col",
    )

    # Ask with both recent and relevant history passed
    stream_gen, citations, docs = service.ask_stream(
        question="Follow up question",
        user_id=42,
        k=2,
        custom_llm=fake_llm,
        recent_history="User: Earlier question\nAssistant: Earlier answer",
        relevant_history="User: Older topic details",
    )

    full_answer = "".join(list(stream_gen))
    assert full_answer == "Answer incorporating history and document."
