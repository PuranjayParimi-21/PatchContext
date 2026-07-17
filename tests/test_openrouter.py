import os
import pytest
from unittest.mock import MagicMock, patch
from app.config import settings
from app.rag_pipeline import PatchContextRAG


def test_openrouter_settings():
    """Verify that OpenRouter settings are correctly loaded from environment config."""
    assert settings.llm_provider == "openrouter"
    assert settings.openrouter_api_key == os.getenv("OPENROUTER_API_KEY")
    assert settings.openrouter_model == "google/gemini-2.5-flash"


@patch("app.rag_pipeline.HybridRetriever")
@patch("app.rag_pipeline.load_vector_store")
@patch("app.rag_pipeline.get_embeddings")
@patch("app.rag_pipeline.ChatOpenAI")
def test_openrouter_llm_initialization(
    mock_chat_openai,
    mock_get_embeddings,
    mock_load_vector_store,
    mock_hybrid_retriever,
):
    """Verify that PatchContextRAG initializes ChatOpenAI with OpenRouter parameters."""

    mock_db = MagicMock()
    mock_load_vector_store.return_value = MagicMock()

    rag = PatchContextRAG(mock_db)

    mock_chat_openai.assert_called_once_with(
        model="google/gemini-2.5-flash",
        temperature=0.0,
        max_tokens=1024,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )