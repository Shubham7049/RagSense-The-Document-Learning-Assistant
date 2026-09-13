"""LangChain retriever factory enforcing per-user isolation and document filtering."""

from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from config.settings import settings
from ingestion.vector_store import get_vector_store
from utils.logger import logger


def get_rag_retriever(
    user_id: int,
    document_id: str | None = None,
    k: int | None = None,
    persist_dir: Path | str | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> BaseRetriever:
    """Build and return a LangChain retriever strictly bound to the active user.

    Parameters
    ----------
    user_id : int
        Authenticated user ID (enforces strict cross-user isolation).
    document_id : str | None
        Optional document ID to restrict retrieval to a single selected document.
    k : int | None
        Top-k chunks to retrieve (defaults to settings.retrieval_k).
    """
    top_k = k or settings.retrieval_k

    # Construct ChromaDB metadata filter
    if document_id:
        filter_dict: dict[str, Any] = {
            "$and": [
                {"user_id": user_id},
                {"document_id": document_id},
            ]
        }
        logger.info(f"Creating retriever for user #{user_id} filtered to document '{document_id}' (k={top_k})")
    else:
        filter_dict = {"user_id": user_id}
        logger.info(f"Creating retriever for user #{user_id} across all documents (k={top_k})")

    store = get_vector_store(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embeddings=embeddings,
    )

    return store.as_retriever(
        search_kwargs={
            "filter": filter_dict,
            "k": top_k,
        }
    )
