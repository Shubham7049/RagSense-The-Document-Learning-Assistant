"""RAG service orchestrating retrieval, ChatGroq integration, and streaming answer generation."""

from pathlib import Path
from typing import Any, Generator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from config.settings import settings
from core.exceptions import LLMError
from rag.chain import build_rag_chain, extract_source_citations, format_docs
from rag.retriever import get_rag_retriever
from utils.logger import logger


class RAGService:
    """Coordinates retrieval, ChatGroq invocation, and token streaming for RAG."""

    def __init__(
        self,
        groq_api_key: str | None = None,
        model_name: str | None = None,
        persist_dir: Path | str | None = None,
        collection_name: str | None = None,
    ) -> None:
        from config.settings import get_settings, reload_settings
        cfg = get_settings()
        if groq_api_key is not None:
            self.api_key = groq_api_key
        else:
            if not cfg.groq_api_key or cfg.groq_api_key.strip() in {"", "your_groq_api_key_here"}:
                cfg = reload_settings()
            self.api_key = cfg.groq_api_key

        self.model_name = model_name or cfg.groq_model
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def get_llm(self, streaming: bool = True, temperature: float = 0.1) -> BaseChatModel:
        """Instantiate and return a ChatGroq instance with credentials check."""
        if not self.api_key or not self.api_key.strip() or self.api_key.strip() == "your_groq_api_key_here":
            logger.warning("Groq API key is missing or default placeholder.")
            raise LLMError(
                "Groq API key is not configured. Please add your GROQ_API_KEY to the .env file."
            )

        try:
            return ChatGroq(
                api_key=self.api_key.strip(),
                model_name=self.model_name,
                temperature=temperature,
                streaming=streaming,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq model '{self.model_name}': {e}")
            raise LLMError(f"Could not connect to Groq LLM: {e}") from e

    def ask_stream(
        self,
        question: str,
        user_id: int,
        document_id: str | None = None,
        k: int | None = None,
        custom_llm: BaseChatModel | None = None,
        recent_history: str = "None",
        relevant_history: str = "None",
    ) -> tuple[Generator[str, None, None], list[dict], list[Document]]:
        """Execute RAG question-answering with progressive token streaming.

        Returns
        -------
        stream_generator : Generator[str, None, None]
            Yields text chunks progressively as they arrive from the LLM.
        citations : list[dict]
            Source metadata extracted from retrieved chunks.
        retrieved_docs : list[Document]
            Raw LangChain Document chunks.
        """
        clean_q = question.strip()
        if not clean_q:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve relevant chunks scoped strictly to user_id
        retriever = get_rag_retriever(
            user_id=user_id,
            document_id=document_id,
            k=k,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )

        try:
            docs = retriever.invoke(clean_q)
            logger.info(f"Retrieved {len(docs)} relevant chunk(s) for user #{user_id} query: '{clean_q[:50]}'")
        except Exception as e:
            logger.error(f"Retrieval error for user #{user_id}: {e}", exc_info=True)
            raise LLMError(f"Document retrieval failed: {e}") from e

        # 2. Extract source citations & format context
        citations = extract_source_citations(docs)
        context_str = format_docs(docs)

        # 3. Resolve LLM (or test mock)
        llm = custom_llm or self.get_llm(streaming=True)

        # 4. Build LCEL chain
        chain = build_rag_chain(llm)

        # 5. Create streaming generator
        def stream_generator() -> Generator[str, None, None]:
            try:
                for chunk in chain.stream(
                    {
                        "context": context_str,
                        "question": clean_q,
                        "recent_history": recent_history,
                        "relevant_history": relevant_history,
                    }
                ):
                    yield chunk
            except Exception as e:
                logger.error(f"Error during response streaming: {e}", exc_info=True)
                yield f"\n\n**Streaming Error**: {e}"

        return stream_generator(), citations, docs

    def ask(
        self,
        question: str,
        user_id: int,
        document_id: str | None = None,
        k: int | None = None,
        custom_llm: BaseChatModel | None = None,
        recent_history: str = "None",
        relevant_history: str = "None",
    ) -> tuple[str, list[dict], list[Document]]:
        """Non-streaming execution of the RAG pipeline."""
        stream_gen, citations, docs = self.ask_stream(
            question=question,
            user_id=user_id,
            document_id=document_id,
            k=k,
            custom_llm=custom_llm,
            recent_history=recent_history,
            relevant_history=relevant_history,
        )
        answer = "".join(list(stream_gen))
        return answer, citations, docs
