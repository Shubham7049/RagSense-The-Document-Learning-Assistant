"""LangChain document loader selection and execution layer."""

from pathlib import Path
from typing import Set

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from core.exceptions import DocumentProcessingError
from utils.logger import logger

# Supported file extensions mapped to description
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "Portable Document Format",
    ".docx": "Microsoft Word Document",
    ".txt": "Plain Text Document",
    ".csv": "Comma-Separated Values",
    ".xlsx": "Microsoft Excel Spreadsheet",
    ".xls": "Legacy Excel Spreadsheet",
    ".json": "JavaScript Object Notation",
}


def get_supported_extensions() -> Set[str]:
    """Return set of lowercased supported file extensions."""
    return set(SUPPORTED_EXTENSIONS.keys())


def is_extension_supported(extension: str) -> bool:
    """Check whether a given file extension is supported."""
    clean_ext = extension.strip().lower()
    if not clean_ext.startswith("."):
        clean_ext = f".{clean_ext}"
    return clean_ext in SUPPORTED_EXTENSIONS


def get_loader_for_file(file_path: Path) -> BaseLoader:
    """Select and instantiate the appropriate modern LangChain loader based on file extension.

    Raises DocumentProcessingError if file format is unsupported or file does not exist.
    """
    if not file_path.exists():
        raise DocumentProcessingError(f"File not found on disk: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path=str(file_path))

    elif ext == ".docx":
        return Docx2txtLoader(file_path=str(file_path))

    elif ext == ".txt":
        return TextLoader(file_path=str(file_path), encoding="utf-8")

    elif ext == ".csv":
        return CSVLoader(file_path=str(file_path), encoding="utf-8")

    elif ext in {".xlsx", ".xls"}:
        return UnstructuredExcelLoader(file_path=str(file_path), mode="elements")

    elif ext == ".json":
        # Load complete JSON structures safely using root jq schema
        return JSONLoader(
            file_path=str(file_path),
            jq_schema=".",
            text_content=False,
        )

    else:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise DocumentProcessingError(
            f"Unsupported file extension '{ext}'. Supported formats: {supported_str}"
        )


def load_documents(file_path: Path) -> list[Document]:
    """Load documents using the designated LangChain loader.

    Returns a list of LangChain Document objects.
    """
    try:
        loader = get_loader_for_file(file_path)
        logger.info(f"Loading '{file_path.name}' using loader: {loader.__class__.__name__}")
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} document element(s) from '{file_path.name}'")
        return docs
    except DocumentProcessingError:
        raise
    except Exception as e:
        logger.error(f"Error executing LangChain loader on '{file_path.name}': {e}", exc_info=True)
        raise DocumentProcessingError(f"Failed to parse document '{file_path.name}': {e}") from e
