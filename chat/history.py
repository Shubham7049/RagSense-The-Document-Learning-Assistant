"""LangChain vector store integration for semantic chat history retrieval."""

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from chat.models import ChatMessage
from config.settings import settings
from ingestion.vector_store import get_vector_store
from utils.logger import logger


def index_chat_message(
    message: ChatMessage,
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> None:
    """Index an individual chat message into the ChromaDB chat history collection."""
    col = collection_name or settings.chroma_chat_collection_name
    doc = Document(
        page_content=message.content,
        metadata={
            "user_id": message.user_id,
            "chat_id": message.chat_id,
            "message_id": message.id,
            "role": message.role,
            "created_at": message.created_at,
        },
    )

    try:
        store = get_vector_store(
            persist_dir=persist_dir,
            collection_name=col,
            embeddings=embeddings,
        )
        store.add_documents([doc], ids=[message.id])
        logger.debug(f"Indexed chat message {message.id} into ChromaDB collection '{col}'")
    except Exception as e:
        logger.warning(f"Could not index chat message {message.id} into vector history: {e}")


def retrieve_relevant_older_history(
    user_id: int,
    chat_id: str,
    query: str,
    exclude_message_ids: set[str],
    k: int = 3,
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> list[Document]:
    """Retrieve semantically relevant older messages for a chat, strictly scoped by user_id and chat_id."""
    col = collection_name or settings.chroma_chat_collection_name
    filter_dict = {
        "$and": [
            {"user_id": user_id},
            {"chat_id": chat_id},
        ]
    }

    try:
        store = get_vector_store(
            persist_dir=persist_dir,
            collection_name=col,
            embeddings=embeddings,
        )
        # Fetch extra candidates to account for recent excluded messages
        fetch_k = k + len(exclude_message_ids) + 2
        candidates = store.similarity_search(query, k=fetch_k, filter=filter_dict)

        relevant = []
        for doc in candidates:
            msg_id = doc.metadata.get("message_id")
            if msg_id and msg_id in exclude_message_ids:
                continue
            relevant.append(doc)
            if len(relevant) >= k:
                break

        logger.debug(f"Retrieved {len(relevant)} relevant older message(s) for chat {chat_id} (user #{user_id})")
        return relevant
    except Exception as e:
        logger.warning(f"Vector chat history retrieval failed: {e}")
        return []


def delete_chat_vectors(
    chat_id: str,
    user_id: int,
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
) -> None:
    """Purge all indexed message vectors for a specific chat and user."""
    col = collection_name or settings.chroma_chat_collection_name
    try:
        store = get_vector_store(
            persist_dir=persist_dir,
            collection_name=col,
        )
        store._collection.delete(
            where={"$and": [{"chat_id": chat_id}, {"user_id": user_id}]}
        )
        logger.info(f"Purged chat history vectors for chat {chat_id} (user #{user_id})")
    except Exception as e:
        logger.warning(f"Error purging chat history vectors for chat {chat_id}: {e}")


def format_messages_for_prompt(messages: Sequence[ChatMessage | Document]) -> str:
    """Format a sequence of ChatMessages or LangChain Documents into a clean dialog block."""
    if not messages:
        return "None"

    formatted = []
    for item in messages:
        if isinstance(item, ChatMessage):
            role = "User" if item.role == "user" else "Assistant"
            formatted.append(f"{role}: {item.content.strip()}")
        elif isinstance(item, Document):
            role_val = item.metadata.get("role", "user")
            role = "User" if role_val == "user" else "Assistant"
            formatted.append(f"{role}: {item.page_content.strip()}")

    return "\n".join(formatted) if formatted else "None"
