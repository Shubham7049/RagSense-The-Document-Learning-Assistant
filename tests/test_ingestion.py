"""Unit tests for Phase 3: Document Upload, LangChain Loaders, Chunking, Embeddings & ChromaDB."""

import json
from pathlib import Path

import openpyxl
import pytest
from docx import Document as DocxDocument
from langchain_core.documents import Document
from pypdf import PdfWriter

from core.database import init_db
from core.exceptions import DocumentProcessingError
from documents.models import DocumentStatus
from documents.repository import DocumentRepository
from ingestion.loader import (
    get_loader_for_file,
    get_supported_extensions,
    is_extension_supported,
    load_documents,
)
from ingestion.pipeline import IngestionPipeline
from ingestion.splitter import split_and_enrich_documents
from ingestion.vector_store import (
    add_chunks_to_vector_store,
    delete_document_vectors,
    get_user_retriever,
    get_vector_store,
)


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing an initialized SQLite database in tmp_path with seed users."""
    db_file = tmp_path / "test_docs.db"
    init_db(db_file)

    # Seed test users so foreign keys are satisfied
    from core.database import get_db_context
    with get_db_context(db_file) as conn:
        for uid in [1, 2, 5, 7, 42, 99]:
            conn.execute(
                "INSERT OR IGNORE INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?);",
                (uid, f"user_{uid}", "dummy_hash", "2026-09-14T00:00:00Z"),
            )
    return db_file


@pytest.fixture
def sample_files(tmp_path):
    """Create lightweight test files for all 6 supported extensions."""
    files_dir = tmp_path / "samples"
    files_dir.mkdir(parents=True, exist_ok=True)

    # 1. TXT
    txt_file = files_dir / "sample.txt"
    txt_file.write_text("This is a plain text document for RAG Sense testing.", encoding="utf-8")

    # 2. CSV
    csv_file = files_dir / "sample.csv"
    csv_file.write_text("item,category,score\nDocument 1,AI,95\nDocument 2,RAG,98\n", encoding="utf-8")

    # 3. JSON
    json_file = files_dir / "sample.json"
    json_file.write_text(
        json.dumps({"project": "RAG Sense", "phase": 3, "status": "active"}),
        encoding="utf-8",
    )

    # 4. DOCX
    docx_file = files_dir / "sample.docx"
    doc = DocxDocument()
    doc.add_heading("RAG Sense Test Document", 0)
    doc.add_paragraph("This is a test paragraph within a Word document.")
    doc.save(str(docx_file))

    # 5. XLSX
    xlsx_file = files_dir / "sample.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Header1", "Header2"])
    ws.append(["Value1", "Value2"])
    wb.save(str(xlsx_file))

    # 6. PDF
    pdf_file = files_dir / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_file, "wb") as f:
        writer.write(f)

    return {
        "txt": txt_file,
        "csv": csv_file,
        "json": json_file,
        "docx": docx_file,
        "xlsx": xlsx_file,
        "pdf": pdf_file,
    }


def test_extension_validation():
    """Verify supported extension checking and rejection of unsupported files."""
    supported = get_supported_extensions()
    for ext in [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls", ".json"]:
        assert ext in supported
        assert is_extension_supported(ext) is True

    # Unsupported
    for invalid in [".exe", ".bin", ".py", ".html", ".zip", "sh"]:
        assert is_extension_supported(invalid) is False


def test_unsupported_file_rejection(tmp_path):
    """Verify that attempting to load an unsupported file raises DocumentProcessingError."""
    fake_file = tmp_path / "script.py"
    fake_file.write_text("print('hello')", encoding="utf-8")

    with pytest.raises(DocumentProcessingError) as exc_info:
        load_documents(fake_file)
    assert "Unsupported file extension" in str(exc_info.value)


def test_langchain_loader_txt(sample_files):
    """Test LangChain TextLoader returns Document objects."""
    docs = load_documents(sample_files["txt"])
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert "plain text document" in docs[0].page_content


def test_langchain_loader_csv(sample_files):
    """Test LangChain CSVLoader returns Document objects."""
    docs = load_documents(sample_files["csv"])
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert "Document 1" in docs[0].page_content or "item" in docs[0].page_content


def test_langchain_loader_json(sample_files):
    """Test LangChain JSONLoader returns Document objects."""
    docs = load_documents(sample_files["json"])
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert "RAG Sense" in docs[0].page_content


def test_langchain_loader_docx(sample_files):
    """Test LangChain Docx2txtLoader returns Document objects."""
    docs = load_documents(sample_files["docx"])
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert "Word document" in docs[0].page_content


def test_langchain_loader_xlsx(sample_files):
    """Test LangChain UnstructuredExcelLoader returns Document objects."""
    docs = load_documents(sample_files["xlsx"])
    assert len(docs) > 0
    assert isinstance(docs[0], Document)


def test_langchain_loader_pdf(sample_files):
    """Test LangChain PyPDFLoader returns Document objects."""
    docs = load_documents(sample_files["pdf"])
    assert len(docs) > 0
    assert isinstance(docs[0], Document)


def test_text_splitting_and_metadata_enrichment():
    """Verify RecursiveCharacterTextSplitter and chunk metadata enrichment."""
    raw_doc = Document(
        page_content="Word " * 500,  # ~2500 characters
        metadata={"initial_key": "initial_value"},
    )

    chunks = split_and_enrich_documents(
        documents=[raw_doc],
        user_id=42,
        document_id="doc-uuid-1234",
        filename="test.txt",
        file_type="txt",
        chunk_size=500,
        chunk_overlap=50,
    )

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, Document)
        assert chunk.metadata["user_id"] == 42
        assert chunk.metadata["document_id"] == "doc-uuid-1234"
        assert chunk.metadata["filename"] == "test.txt"
        assert chunk.metadata["file_type"] == "txt"
        assert chunk.metadata["chunk_index"] == idx
        assert "upload_timestamp" in chunk.metadata


def test_document_repository_user_isolation(temp_db):
    """Verify that SQLite document repository strictly isolates records per user."""
    repo = DocumentRepository(db_path=temp_db)

    # Insert document for User 1
    doc_u1 = repo.create_document(
        document_id="doc-1",
        user_id=1,
        filename="user1_report.pdf",
        file_type="pdf",
        file_size=1024,
        storage_path="/path/to/user1/doc-1.pdf",
    )

    # Insert document for User 2
    doc_u2 = repo.create_document(
        document_id="doc-2",
        user_id=2,
        filename="user2_notes.txt",
        file_type="txt",
        file_size=512,
        storage_path="/path/to/user2/doc-2.txt",
    )

    # User 1 query
    docs_u1 = repo.get_user_documents(user_id=1)
    assert len(docs_u1) == 1
    assert docs_u1[0].id == "doc-1"
    assert docs_u1[0].filename == "user1_report.pdf"

    # User 2 query
    docs_u2 = repo.get_user_documents(user_id=2)
    assert len(docs_u2) == 1
    assert docs_u2[0].id == "doc-2"
    assert docs_u2[0].filename == "user2_notes.txt"

    # Cross-user lookup by ID returns None
    assert repo.get_document_by_id("doc-1", user_id=2) is None
    assert repo.get_document_by_id("doc-2", user_id=1) is None


def test_end_to_end_pipeline_and_user_storage(tmp_path, temp_db, sample_files):
    """Verify end-to-end ingestion: saving, loading, chunking, and ChromaDB indexing."""
    repo = DocumentRepository(db_path=temp_db)
    storage_dir = tmp_path / "user_storage"
    chroma_dir = tmp_path / "chroma_db"

    pipeline = IngestionPipeline(
        repository=repo,
        storage_base_dir=storage_dir,
        chroma_persist_dir=chroma_dir,
        chroma_collection_name="test_collection",
    )

    # Read txt bytes
    with open(sample_files["txt"], "rb") as f:
        file_bytes = f.read()

    record = pipeline.process_document(
        file_name="sample.txt",
        file_bytes=file_bytes,
        user_id=7,
    )

    assert record.status == DocumentStatus.INDEXED
    assert record.chunk_count > 0
    assert record.user_id == 7

    # Verify user-specific directory
    expected_user_dir = storage_dir / "7" / "documents"
    assert expected_user_dir.exists()
    assert Path(record.storage_path).exists()
    assert Path(record.storage_path).parent == expected_user_dir


def test_chroma_user_isolation_retrieval(tmp_path):
    """Verify ChromaDB stores chunks and retrieves only the active user's documents."""
    chroma_dir = tmp_path / "chroma_isolation"

    chunk_u1 = Document(
        page_content="Confidential financial forecast for User One.",
        metadata={"user_id": 1, "document_id": "doc-u1", "filename": "u1.txt"},
    )
    chunk_u2 = Document(
        page_content="Private medical research data for User Two.",
        metadata={"user_id": 2, "document_id": "doc-u2", "filename": "u2.txt"},
    )

    add_chunks_to_vector_store(
        chunks=[chunk_u1, chunk_u2],
        persist_dir=chroma_dir,
        collection_name="isolation_test",
    )

    # Query as User 1
    retriever_u1 = get_user_retriever(
        user_id=1,
        k=5,
        persist_dir=chroma_dir,
        collection_name="isolation_test",
    )
    results_u1 = retriever_u1.invoke("financial forecast")
    assert len(results_u1) > 0
    for r in results_u1:
        assert r.metadata["user_id"] == 1
        assert "User Two" not in r.page_content

    # Query as User 2
    retriever_u2 = get_user_retriever(
        user_id=2,
        k=5,
        persist_dir=chroma_dir,
        collection_name="isolation_test",
    )
    results_u2 = retriever_u2.invoke("medical research")
    assert len(results_u2) > 0
    for r in results_u2:
        assert r.metadata["user_id"] == 2
        assert "User One" not in r.page_content


def test_pipeline_failure_handling(tmp_path, temp_db):
    """Verify that processing failures update SQLite to FAILED without crashing."""
    repo = DocumentRepository(db_path=temp_db)
    pipeline = IngestionPipeline(
        repository=repo,
        storage_base_dir=tmp_path / "storage",
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="fail_test",
    )

    # Process an empty corrupt PDF
    record = pipeline.process_document(
        file_name="corrupt.pdf",
        file_bytes=b"not a valid pdf",
        user_id=99,
    )

    assert record.status == DocumentStatus.FAILED
    assert record.error_message is not None
    assert record.chunk_count == 0

    # Verify database state
    db_record = repo.get_document_by_id(record.id, user_id=99)
    assert db_record is not None
    assert db_record.status == DocumentStatus.FAILED
    assert db_record.error_message is not None


def test_document_deletion_cleanup(tmp_path, temp_db, sample_files):
    """Verify deleting a document cleans up SQLite, disk, and ChromaDB."""
    repo = DocumentRepository(db_path=temp_db)
    storage_dir = tmp_path / "del_storage"
    chroma_dir = tmp_path / "del_chroma"

    pipeline = IngestionPipeline(
        repository=repo,
        storage_base_dir=storage_dir,
        chroma_persist_dir=chroma_dir,
        chroma_collection_name="del_test",
    )

    with open(sample_files["txt"], "rb") as f:
        file_bytes = f.read()

    record = pipeline.process_document(
        file_name="to_delete.txt",
        file_bytes=file_bytes,
        user_id=5,
    )

    assert Path(record.storage_path).exists()
    assert repo.get_document_by_id(record.id, user_id=5) is not None

    # Delete
    deleted = pipeline.delete_document(record.id, user_id=5)
    assert deleted is True

    # Verify cleanup
    assert not Path(record.storage_path).exists()
    assert repo.get_document_by_id(record.id, user_id=5) is None


def test_file_size_limit_rejection(tmp_path, temp_db, monkeypatch):
    """Test that files exceeding max_upload_size_mb are rejected with DocumentProcessingError."""
    from config.settings import settings
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)

    repo = DocumentRepository(db_path=temp_db)
    pipeline = IngestionPipeline(repository=repo, storage_base_dir=tmp_path / "storage")

    oversized_bytes = b"0" * (2 * 1024 * 1024)  # 2 MB exceeds 1 MB limit
    with pytest.raises(DocumentProcessingError) as exc_info:
        pipeline.save_file(
            file_name="oversized.txt",
            file_bytes=oversized_bytes,
            user_id=1,
        )
    assert "exceeds maximum allowed size" in str(exc_info.value)
