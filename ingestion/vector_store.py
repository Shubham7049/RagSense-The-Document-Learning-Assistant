"""LangChain ChromaDB vector store integration with strict user isolation."""

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from core.exceptions import VectorStoreError
from ingestion.embeddings import get_embeddings
from utils.logger import logger


def get_vector_store(
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
):
    """Instantiate and return the persistent LangChain Chroma vector store."""
    target_dir = Path(persist_dir) if persist_dir else settings.chroma_persist_directory
    target_dir.mkdir(parents=True, exist_ok=True)
    target_collection = collection_name or settings.chroma_collection_name
    emb = embeddings or get_embeddings()

    try:
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=target_collection,
            embedding_function=emb,
            persist_directory=str(target_dir),
        )
    except ImportError:
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name=target_collection,
            embedding_function=emb,
            persist_directory=str(target_dir),
        )


def add_chunks_to_vector_store(
    chunks: Sequence[Document],
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> list[str]:
    """Index a list of enriched document chunks into persistent ChromaDB.

    Every chunk is verified to contain 'user_id' and 'document_id' in its metadata.
    """
    if not chunks:
        return []

    # Verification: every chunk must have user_id and document_id
    for idx, chunk in enumerate(chunks):
        if "user_id" not in chunk.metadata or "document_id" not in chunk.metadata:
            raise VectorStoreError(
                f"Chunk #{idx} missing required isolation metadata ('user_id' or 'document_id')"
            )

    try:
        store = get_vector_store(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        doc_ids = store.add_documents(list(chunks))
        logger.info(f"Indexed {len(doc_ids)} chunk(s) into ChromaDB collection '{store._collection.name}'")
        return doc_ids
    except Exception as e:
        logger.error(f"Failed to add chunks to ChromaDB: {e}", exc_info=True)
        raise VectorStoreError(f"Vector store indexing failed: {e}") from e


def delete_document_vectors(
    document_id: str,
    user_id: int,
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> None:
    """Delete all vector embeddings belonging to a specific document and user."""
    try:
        store = get_vector_store(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        # Chroma where filter with composite and
        store._collection.delete(
            where={"$and": [{"document_id": document_id}, {"user_id": user_id}]}
        )
        logger.info(f"Deleted vector chunks for document '{document_id}' (user #{user_id})")
    except Exception as e:
        logger.warning(f"Error deleting vectors for document '{document_id}': {e}")


def get_user_retriever(
    user_id: int,
    k: int = 4,
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
):
    """Return a LangChain retriever strictly scoped to the active user's document chunks."""
    store = get_vector_store(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    return store.as_retriever(
        search_kwargs={
            "filter": {"user_id": user_id},
            "k": k,
        }
    )
