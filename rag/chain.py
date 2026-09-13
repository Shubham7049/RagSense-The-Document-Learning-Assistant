"""LangChain RAG chain composition using LCEL."""

from typing import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag.prompts import get_rag_prompt_template


def format_docs(docs: Sequence[Document]) -> str:
    """Format a list of LangChain Document chunks into an annotated context block."""
    if not docs:
        return "No relevant context found in documents."

    formatted = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "Unknown Source")
        chunk_idx = doc.metadata.get("chunk_index", "N/A")
        content = doc.page_content.strip()
        formatted.append(f"--- [Snippet {idx}: {source} | Chunk #{chunk_idx}] ---\n{content}")

    return "\n\n".join(formatted)


def extract_source_citations(docs: Sequence[Document]) -> list[dict]:
    """Extract clean source references with page numbers from retrieved document chunks."""
    citations = []
    for idx, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        snippet = doc.page_content.strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        raw_page = metadata.get("page")
        if isinstance(raw_page, int):
            page_number = raw_page + 1
        elif "page_number" in metadata and isinstance(metadata["page_number"], int):
            page_number = metadata["page_number"]
        else:
            page_number = None

        citations.append(
            {
                "index": idx,
                "filename": metadata.get("filename", "Unknown Document"),
                "document_id": metadata.get("document_id", "Unknown"),
                "file_type": metadata.get("file_type", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "page_number": page_number,
                "source": metadata.get("source", ""),
                "snippet": snippet,
            }
        )
    return citations


def build_rag_chain(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate | None = None,
) -> Runnable:
    """Construct a modern LangChain LCEL chain: {context, question} | prompt | llm | StrOutputParser."""
    rag_prompt = prompt or get_rag_prompt_template()
    return rag_prompt | llm | StrOutputParser()
