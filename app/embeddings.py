import logging
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from app.config import settings

logger = logging.getLogger("PatchContext.Embeddings")

def get_embeddings() -> Embeddings:
    """Initializes and returns the appropriate embedding model based on settings."""
    model_name = settings.embedding_model if settings.embedding_model else "text-embedding-ada-002"
    
    if settings.llm_provider.lower() == "openrouter" and settings.openrouter_api_key:
        logger.info(f"Initializing OpenAI Embeddings ({model_name}) via OpenRouter...")
        model_id = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
        return OpenAIEmbeddings(
            model=model_id,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
    else:
        logger.info(f"Initializing OpenAI Embeddings ({model_name})...")
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set. Embedding calls will fail.")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openai_api_key
        )
