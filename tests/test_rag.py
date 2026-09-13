"""Unit tests for Phase 4: LangChain RAG Question Answering, Groq integration & Streaming."""

from unittest.mock import MagicMock

import pytest
from langchain_community.chat_models import FakeListChatModel
from langchain_core.documents import Document

from core.exceptions import LLMError
from ingestion.vector_store import add_chunks_to_vector_store
from rag.chain import build_rag_chain, extract_source_citations, format_docs
from rag.prompts import get_rag_prompt_template
from rag.retriever import get_rag_retriever
from rag.service import RAGService


def test_rag_prompt_template():
    """Verify LangChain ChatPromptTemplate contains required instructional clauses."""
    prompt = get_rag_prompt_template()
    formatted = prompt.format(context="Sample document text.", question="What is this?")

    assert "RAG Sense" in formatted
    assert "Sample document text." in formatted
    assert "What is this?" in formatted
    assert "Zero Hallucination" in formatted


def test_format_docs_and_citations():
    """Test context formatting and citation metadata extraction from documents."""
    # Empty docs
    assert "No relevant context" in format_docs([])
    assert extract_source_citations([]) == []

    # Non-empty docs
    doc1 = Document(
        page_content="Artificial Intelligence overview.",
        metadata={"filename": "ai_intro.pdf", "document_id": "doc-1", "chunk_index": 0},
    )
    doc2 = Document(
        page_content="Machine Learning basics.",
        metadata={"filename": "ml_notes.txt", "document_id": "doc-2", "chunk_index": 3},
    )

    formatted = format_docs([doc1, doc2])
    assert "ai_intro.pdf" in formatted
    assert "ml_notes.txt" in formatted
    assert "Chunk #0" in formatted
    assert "Chunk #3" in formatted

    citations = extract_source_citations([doc1, doc2])
    assert len(citations) == 2
    assert citations[0]["filename"] == "ai_intro.pdf"
    assert citations[0]["document_id"] == "doc-1"
    assert citations[0]["chunk_index"] == 0
    assert "Artificial Intelligence" in citations[0]["snippet"]


def test_retriever_filter_construction(tmp_path):
    """Test that retriever strictly applies user isolation and optional document filters."""
    chroma_dir = tmp_path / "chroma_retriever_test"

    # User-only filter
    retriever_user = get_rag_retriever(
        user_id=101,
        k=3,
        persist_dir=chroma_dir,
        collection_name="retriever_test",
    )
    search_kwargs = retriever_user.search_kwargs
    assert search_kwargs["filter"] == {"user_id": 101}
    assert search_kwargs["k"] == 3

    # User + Document filter
    retriever_doc = get_rag_retriever(
        user_id=101,
        document_id="doc-xyz",
        k=5,
        persist_dir=chroma_dir,
        collection_name="retriever_test",
    )
    doc_filter = retriever_doc.search_kwargs["filter"]
    assert "$and" in doc_filter
    assert {"user_id": 101} in doc_filter["$and"]
    assert {"document_id": "doc-xyz"} in doc_filter["$and"]


def test_lcel_rag_chain_streaming():
    """Test LangChain LCEL RAG chain execution and token streaming with a mock chat model."""
    fake_llm = FakeListChatModel(responses=["RAG Sense is an educational assistant."])
    chain = build_rag_chain(fake_llm)

    # Invoke
    result = chain.invoke({"context": "Sample context", "question": "What is RAG Sense?"})
    assert result == "RAG Sense is an educational assistant."

    # Stream
    fake_stream_llm = FakeListChatModel(responses=["Streamed ", "response ", "tokens."])
    stream_chain = build_rag_chain(fake_stream_llm)
    stream_output = list(stream_chain.stream({"context": "Context", "question": "Question"}))
    assert len(stream_output) > 0


def test_rag_service_with_mock_llm(tmp_path):
    """Test full RAGService streaming pipeline with indexed ChromaDB chunks and mock LLM."""
    chroma_dir = tmp_path / "rag_service_chroma"

    # Index chunks for User 77
    chunk = Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={
            "user_id": 77,
            "document_id": "doc-77",
            "filename": "langchain_guide.pdf",
            "file_type": "pdf",
            "chunk_index": 0,
        },
    )
    add_chunks_to_vector_store(
        chunks=[chunk],
        persist_dir=chroma_dir,
        collection_name="service_test",
    )

    fake_llm = FakeListChatModel(responses=["LangChain is an LLM framework."])
    service = RAGService(
        groq_api_key="gsk_dummy_key",
        persist_dir=chroma_dir,
        collection_name="service_test",
    )

    # Ask with custom LLM
    stream_gen, citations, docs = service.ask_stream(
        question="What is LangChain?",
        user_id=77,
        k=3,
        custom_llm=fake_llm,
    )

    streamed_tokens = list(stream_gen)
    full_answer = "".join(streamed_tokens)

    assert "LangChain is an LLM framework" in full_answer
    assert len(citations) == 1
    assert citations[0]["filename"] == "langchain_guide.pdf"
    assert citations[0]["document_id"] == "doc-77"


def test_rag_service_user_isolation(tmp_path):
    """Verify that User A cannot retrieve User B's documents through the RAG pipeline."""
    chroma_dir = tmp_path / "isolation_chroma"

    # User 88 document
    chunk_88 = Document(
        page_content="User 88's confidential internal salary report.",
        metadata={"user_id": 88, "document_id": "doc-88", "filename": "salary.pdf"},
    )
    add_chunks_to_vector_store(
        chunks=[chunk_88],
        persist_dir=chroma_dir,
        collection_name="rag_isolation",
    )

    fake_llm = FakeListChatModel(responses=["No information available."])
    service = RAGService(
        groq_api_key="gsk_dummy_key",
        persist_dir=chroma_dir,
        collection_name="rag_isolation",
    )

    # User 99 (different user) asks about salary
    stream_gen, citations, docs = service.ask_stream(
        question="What are the salaries?",
        user_id=99,
        k=3,
        custom_llm=fake_llm,
    )

    # Must retrieve 0 chunks and 0 citations for User 99
    assert len(docs) == 0
    assert len(citations) == 0


def test_missing_groq_api_key_handling():
    """Verify that missing or default Groq API keys raise a clean LLMError without crashing."""
    service_no_key = RAGService(groq_api_key="")
    with pytest.raises(LLMError) as exc_info:
        service_no_key.get_llm()
    assert "Groq API key is not configured" in str(exc_info.value)

    service_placeholder = RAGService(groq_api_key="your_groq_api_key_here")
    with pytest.raises(LLMError) as exc_info2:
        service_placeholder.get_llm()
    assert "Groq API key is not configured" in str(exc_info2.value)


def test_empty_question_handling():
    """Verify that empty or whitespace-only questions raise ValueError."""
    service = RAGService(groq_api_key="gsk_dummy")
    with pytest.raises(ValueError) as exc_info:
        service.ask_stream(question="   ", user_id=1)
    assert "Question cannot be empty" in str(exc_info.value)
