import logging
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings

logger = logging.getLogger("PatchContext.Embeddings")

def get_embeddings() -> Embeddings:
    """Initializes and returns the appropriate embedding model based on settings."""
    provider = settings.embedding_provider.lower()
    
    if provider == "local":
        logger.info("Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        logger.info("Initializing OpenAI Embeddings (text-embedding-3-small)...")
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set. Embedding calls will fail.")
            
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
