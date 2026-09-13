"""LangChain embeddings integration using Sentence Transformers."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config.settings import settings
from utils.logger import logger


@lru_cache(maxsize=1)
def get_embeddings(model_name: str | None = None) -> Embeddings:
    """Instantiate and cache the LangChain Sentence Transformer embeddings model.

    Uses the model specified in settings (default: all-MiniLM-L6-v2).
    """
    target_model = model_name or settings.embedding_model
    logger.info(f"Initializing LangChain HuggingFaceEmbeddings with model: '{target_model}'")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=target_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        # Fallback to community integration if langchain-huggingface is loading
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=target_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
