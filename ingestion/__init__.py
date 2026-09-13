"""Ingestion package for RAG Sense."""

from ingestion.embeddings import get_embeddings
from ingestion.loader import (
    SUPPORTED_EXTENSIONS,
    get_loader_for_file,
    get_supported_extensions,
    is_extension_supported,
    load_documents,
)
from ingestion.pipeline import IngestionPipeline
from ingestion.splitter import get_text_splitter, split_and_enrich_documents
from ingestion.vector_store import (
    add_chunks_to_vector_store,
    delete_document_vectors,
    get_user_retriever,
    get_vector_store,
)

__all__ = [
    "IngestionPipeline",
    "SUPPORTED_EXTENSIONS",
    "add_chunks_to_vector_store",
    "delete_document_vectors",
    "get_embeddings",
    "get_loader_for_file",
    "get_supported_extensions",
    "get_text_splitter",
    "get_user_retriever",
    "get_vector_store",
    "is_extension_supported",
    "load_documents",
    "split_and_enrich_documents",
]
