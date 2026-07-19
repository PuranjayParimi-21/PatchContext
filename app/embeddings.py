import logging
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings

logger = logging.getLogger("PatchContext.Embeddings")

def get_embeddings() -> Embeddings:
    """Initializes and returns the appropriate embedding model based on settings."""
    provider = settings.embedding_provider.lower() if settings.embedding_provider else "openai"
    model_name = settings.embedding_model if settings.embedding_model else "text-embedding-ada-002"
    
    if provider in ["local", "huggingface"]:
        # Fallback to a valid HuggingFace model if the model is set to an OpenAI model name
        if "text-embedding-" in model_name or "ada" in model_name:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Initializing local HuggingFace Embeddings ({model_name})...")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        
    if settings.llm_provider.lower() == "openrouter" and settings.openrouter_api_key:
        logger.info(f"Initializing OpenAI Embeddings ({model_name}) via OpenRouter...")
        model_id = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
        return OpenAIEmbeddings(
            model=model_id,
            openai_api_key=settings.openrouter_api_key,  # type: ignore
            openai_api_base="https://openrouter.ai/api/v1",  # type: ignore
            chunk_size=100
        )
    else:
        logger.info(f"Initializing OpenAI Embeddings ({model_name})...")
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set. Embedding calls will fail.")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openai_api_key,  # type: ignore
            chunk_size=100
        )


