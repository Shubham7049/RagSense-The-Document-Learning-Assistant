"""Ingestion pipeline orchestrating file storage, LangChain loading, chunking, and ChromaDB indexing."""

import uuid
from pathlib import Path
from typing import BinaryIO

from config.settings import settings
from core.exceptions import DocumentProcessingError
from documents.models import DocumentRecord, DocumentStatus
from documents.repository import DocumentRepository
from ingestion.loader import is_extension_supported, load_documents
from ingestion.splitter import split_and_enrich_documents
from ingestion.vector_store import add_chunks_to_vector_store, delete_document_vectors
from utils.logger import logger


class IngestionPipeline:
    """Manages the lifecycle of document upload, validation, LangChain loading, and vector indexing."""

    def __init__(
        self,
        repository: DocumentRepository | None = None,
        storage_base_dir: Path | None = None,
        chroma_persist_dir: Path | str | None = None,
        chroma_collection_name: str | None = None,
    ) -> None:
        self.repository = repository or DocumentRepository()
        self.storage_base_dir = storage_base_dir or settings.document_storage_directory
        self.chroma_persist_dir = chroma_persist_dir or settings.chroma_persist_directory
        self.chroma_collection_name = chroma_collection_name or settings.chroma_collection_name

    def get_user_storage_dir(self, user_id: int) -> Path:
        """Return and ensure the dedicated storage directory for a specific user."""
        user_dir = self.storage_base_dir / str(user_id) / "documents"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def save_file(
        self,
        file_name: str,
        file_bytes: bytes | BinaryIO,
        user_id: int,
        document_id: str | None = None,
    ) -> tuple[Path, str]:
        """Save an uploaded file into the user's isolated storage directory."""
        doc_id = document_id or str(uuid.uuid4())
        ext = Path(file_name).suffix.lower()

        if not is_extension_supported(ext):
            raise DocumentProcessingError(f"Unsupported file format '{ext}'.")

        user_dir = self.get_user_storage_dir(user_id)
        # Unique disk filename prefixed with document_id to avoid collisions
        safe_filename = f"{doc_id}_{Path(file_name).name}"
        dest_path = user_dir / safe_filename

        content = file_bytes.read() if hasattr(file_bytes, "read") else file_bytes
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise DocumentProcessingError(
                f"File '{file_name}' ({len(content) / (1024 * 1024):.1f} MB) exceeds maximum allowed size of {settings.max_upload_size_mb} MB."
            )

        with open(dest_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved file '{file_name}' to user #{user_id} path: {dest_path}")
        return dest_path, doc_id

    def process_document(
        self,
        file_name: str,
        file_bytes: bytes | BinaryIO,
        user_id: int,
    ) -> DocumentRecord:
        """Execute the end-to-end ingestion pipeline for a document.

        1. Validate & save to user-specific filesystem
        2. Record in SQLite (PENDING)
        3. Parse via LangChain loader
        4. Split into chunks via LangChain RecursiveCharacterTextSplitter
        5. Embed and index into ChromaDB
        6. Update SQLite record to INDEXED with chunk count
        """
        ext = Path(file_name).suffix.lower()
        file_type = ext.lstrip(".").lower()
        doc_id = str(uuid.uuid4())

        # 1. Validate & Save
        saved_path, doc_id = self.save_file(
            file_name=file_name,
            file_bytes=file_bytes,
            user_id=user_id,
            document_id=doc_id,
        )
        file_size = saved_path.stat().st_size

        # 2. SQLite record (PENDING)
        record = self.repository.create_document(
            document_id=doc_id,
            user_id=user_id,
            filename=file_name,
            file_type=file_type,
            file_size=file_size,
            storage_path=str(saved_path),
        )

        try:
            # 3. Transition to PROCESSING
            self.repository.update_status(doc_id, DocumentStatus.PROCESSING)

            # 4. Load via LangChain Loader
            raw_docs = load_documents(saved_path)
            if not raw_docs:
                raise DocumentProcessingError(f"Document '{file_name}' yielded no extractable content.")

            # 5. Split via LangChain RecursiveCharacterTextSplitter
            chunks = split_and_enrich_documents(
                documents=raw_docs,
                user_id=user_id,
                document_id=doc_id,
                filename=file_name,
                file_type=file_type,
            )

            # 6. Embed & Store in ChromaDB
            add_chunks_to_vector_store(
                chunks=chunks,
                persist_dir=self.chroma_persist_dir,
                collection_name=self.chroma_collection_name,
            )

            # 7. Update to INDEXED
            self.repository.update_status(
                document_id=doc_id,
                status=DocumentStatus.INDEXED,
                chunk_count=len(chunks),
            )
            record.status = DocumentStatus.INDEXED
            record.chunk_count = len(chunks)
            logger.info(f"Successfully indexed document '{file_name}' ({doc_id}) into ChromaDB.")
            return record

        except Exception as e:
            err_msg = str(e)
            logger.error(f"Failed to process document '{file_name}' ({doc_id}): {err_msg}", exc_info=True)
            self.repository.update_status(
                document_id=doc_id,
                status=DocumentStatus.FAILED,
                chunk_count=0,
                error_message=err_msg,
            )
            record.status = DocumentStatus.FAILED
            record.error_message = err_msg
            return record

    def delete_document(self, document_id: str, user_id: int) -> bool:
        """Remove a document from SQLite, filesystem, and ChromaDB vector store."""
        doc = self.repository.get_document_by_id(document_id, user_id)
        if not doc:
            return False

        # 1. Delete vector embeddings in ChromaDB
        delete_document_vectors(
            document_id=document_id,
            user_id=user_id,
            persist_dir=self.chroma_persist_dir,
            collection_name=self.chroma_collection_name,
        )

        # 2. Delete file on disk
        try:
            file_path = Path(doc.storage_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted physical file for document '{document_id}': {file_path}")
        except Exception as e:
            logger.warning(f"Could not remove physical file '{doc.storage_path}': {e}")

        # 3. Delete database record
        return self.repository.delete_document(document_id, user_id)
