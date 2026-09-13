"""Text splitting and chunk metadata enrichment using LangChain."""

from datetime import datetime, timezone
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import logger


def get_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return a configured LangChain RecursiveCharacterTextSplitter instance."""
    c_size = chunk_size or settings.chunk_size
    c_overlap = chunk_overlap or settings.chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def split_and_enrich_documents(
    documents: Sequence[Document],
    user_id: int,
    document_id: str,
    filename: str,
    file_type: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split LangChain Document objects into chunks and enrich metadata for strict user isolation.

    Every generated chunk is guaranteed to contain:
    - user_id (for cross-user isolation)
    - document_id (for document lifecycle & cleanup)
    - filename & file_type
    - chunk_index
    - upload_timestamp
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    split_chunks = splitter.split_documents(documents)

    now = datetime.now(timezone.utc).isoformat()

    enriched_chunks: list[Document] = []
    for idx, chunk in enumerate(split_chunks):
        # Merge existing metadata while strictly overriding user and document ownership
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "user_id": user_id,
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "source": filename,
                "chunk_index": idx,
                "upload_timestamp": now,
            }
        )
        enriched_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    logger.info(
        f"Split document '{filename}' into {len(enriched_chunks)} chunk(s) "
        f"(user #{user_id}, doc_id: {document_id})"
    )
    return enriched_chunks
