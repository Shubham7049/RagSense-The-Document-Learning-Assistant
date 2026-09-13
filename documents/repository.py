"""SQLite repository for document tracking with user isolation."""

from datetime import datetime, timezone
from pathlib import Path

from core.database import get_db_context
from documents.models import DocumentRecord, DocumentStatus
from utils.logger import logger


class DocumentRepository:
    """Handles SQLite persistence for documents, enforcing per-user isolation."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = db_path

    def create_document(
        self,
        document_id: str,
        user_id: int,
        filename: str,
        file_type: str,
        file_size: int,
        storage_path: str,
    ) -> DocumentRecord:
        """Insert a new document record in PENDING status."""
        now = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO documents (
            id, user_id, filename, file_type, file_size,
            storage_path, upload_timestamp, status, chunk_count, error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL);
        """
        with get_db_context(self.db_path) as conn:
            conn.execute(
                query,
                (
                    document_id,
                    user_id,
                    filename,
                    file_type,
                    file_size,
                    storage_path,
                    now,
                    DocumentStatus.PENDING.value,
                ),
            )

        logger.info(f"Registered document '{filename}' ({document_id}) for user #{user_id}")
        return DocumentRecord(
            id=document_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            storage_path=storage_path,
            upload_timestamp=now,
            status=DocumentStatus.PENDING,
            chunk_count=0,
            error_message=None,
        )

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Update processing status, chunk count, and optional error message."""
        query = """
        UPDATE documents
        SET status = ?, chunk_count = ?, error_message = ?
        WHERE id = ?;
        """
        with get_db_context(self.db_path) as conn:
            conn.execute(query, (status.value, chunk_count, error_message, document_id))

        logger.info(f"Updated document {document_id} status -> {status.value} (chunks: {chunk_count})")

    def get_user_documents(self, user_id: int) -> list[DocumentRecord]:
        """Fetch all documents belonging to the authenticated user (strict isolation)."""
        query = """
        SELECT id, user_id, filename, file_type, file_size,
               storage_path, upload_timestamp, status, chunk_count, error_message
        FROM documents
        WHERE user_id = ?
        ORDER BY upload_timestamp DESC;
        """
        with get_db_context(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            return [
                DocumentRecord(
                    id=row["id"],
                    user_id=row["user_id"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    file_size=row["file_size"],
                    storage_path=row["storage_path"],
                    upload_timestamp=row["upload_timestamp"],
                    status=DocumentStatus(row["status"]),
                    chunk_count=row["chunk_count"],
                    error_message=row["error_message"],
                )
                for row in rows
            ]

    def list_documents(self, user_id: int) -> list[DocumentRecord]:
        """Convenience alias for get_user_documents."""
        return self.get_user_documents(user_id)

    def get_document_by_id(self, document_id: str, user_id: int) -> DocumentRecord | None:
        """Retrieve a specific document, enforcing ownership check."""
        query = """
        SELECT id, user_id, filename, file_type, file_size,
               storage_path, upload_timestamp, status, chunk_count, error_message
        FROM documents
        WHERE id = ? AND user_id = ?;
        """
        with get_db_context(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (document_id, user_id))
            row = cursor.fetchone()
            if not row:
                return None
            return DocumentRecord(
                id=row["id"],
                user_id=row["user_id"],
                filename=row["filename"],
                file_type=row["file_type"],
                file_size=row["file_size"],
                storage_path=row["storage_path"],
                upload_timestamp=row["upload_timestamp"],
                status=DocumentStatus(row["status"]),
                chunk_count=row["chunk_count"],
                error_message=row["error_message"],
            )

    def delete_document(self, document_id: str, user_id: int) -> bool:
        """Delete a document record, constrained by user ID."""
        query = "DELETE FROM documents WHERE id = ? AND user_id = ?;"
        with get_db_context(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (document_id, user_id))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted document record {document_id} for user #{user_id}")
            return deleted
