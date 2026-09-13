"""RAG package for RAG Sense."""

from rag.chain import build_rag_chain, extract_source_citations, format_docs
from rag.prompts import get_rag_prompt_template
from rag.retriever import get_rag_retriever
from rag.service import RAGService

__all__ = [
    "RAGService",
    "build_rag_chain",
    "extract_source_citations",
    "format_docs",
    "get_rag_prompt_template",
    "get_rag_retriever",
]
