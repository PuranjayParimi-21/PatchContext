import os
# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import MagicMock, patch
from app.config import settings
from app.rag_pipeline import PatchContextRAG


def test_openrouter_settings():
    """Verify that OpenRouter settings can be loaded from config."""
    from app.config import Settings
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openrouter",
        "OPENROUTER_API_KEY": "sk-or-v1-test-key",
        "OPENROUTER_MODEL": "google/gemini-2.5-flash"
    }):
        custom_settings = Settings()
        assert custom_settings.llm_provider == "openrouter"
        assert custom_settings.openrouter_api_key == "sk-or-v1-test-key"
        assert custom_settings.openrouter_model == "google/gemini-2.5-flash"


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
    """Verify that PatchContextRAG initializes ChatOpenAI with OpenRouter parameters when provider is openrouter."""
    orig_provider = settings.llm_provider
    orig_key = settings.openrouter_api_key
    orig_model = settings.openrouter_model
    
    settings.llm_provider = "openrouter"
    settings.openrouter_api_key = "sk-or-v1-fake-key"
    settings.openrouter_model = "meta-llama/llama-3.2-3b-instruct:free"
    
    try:
        mock_db = MagicMock()
        mock_load_vector_store.return_value = MagicMock()
        
        rag = PatchContextRAG(mock_db)
        
        mock_chat_openai.assert_called_once_with(
            model="meta-llama/llama-3.2-3b-instruct:free",
            temperature=0.3,
            max_tokens=512,
            openai_api_key="sk-or-v1-fake-key",
            openai_api_base="https://openrouter.ai/api/v1",
        )
    finally:
        settings.llm_provider = orig_provider
        settings.openrouter_api_key = orig_key
        settings.openrouter_model = orig_model