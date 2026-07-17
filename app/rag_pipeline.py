import time
import logging
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from app.config import settings
from app.database import DatabaseManager
from app.vector_store import load_vector_store
from app.embeddings import get_embeddings
from app.retriever import HybridRetriever
from app.verifier import HallucinationGuard
from app.prompt import QA_PROMPT

logger = logging.getLogger("PatchContext.RagPipeline")

class PatchContextRAG:
    """Core RAG pipeline linking retrieval, LLM generation, citation check, and NLI verification."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.embeddings = get_embeddings()
        
        # Load vector store
        self.vectorstore = load_vector_store(self.embeddings)
        
        # Initialize retriever (rebuilds BM25 based on DB)
        if self.vectorstore:
            self.retriever = HybridRetriever(self.db, self.vectorstore)
        else:
            logger.warning("Vector store not found. Hybrid retriever initialization delayed.")
            self.retriever = None
            
        # Initialize hallucination guard
        self.guard = HallucinationGuard(self.db)
        
        # Initialize LLM
        provider = settings.llm_provider.lower()
        if provider == "openrouter":
            logger.info(f"Initializing OpenRouter LLM ({settings.openrouter_model})...")
            if not settings.openrouter_api_key:
                logger.warning("OPENROUTER_API_KEY environment variable is not set. LLM calls will fail.")
            self.llm = ChatOpenAI(
                model=settings.openrouter_model,
                temperature=0.0,
                max_tokens=1024,
                openai_api_key=settings.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1"
            )
        else:
            model_name = settings.model if settings.model else "gpt-4o-mini"
            logger.info(f"Initializing ChatOpenAI ({model_name})...")
            if not settings.openai_api_key:
                logger.warning("OPENAI_API_KEY environment variable is not set. LLM calls will fail.")
                
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                openai_api_key=settings.openai_api_key
            )

    def refresh_retriever(self) -> None:
        """Reloads the vector store and BM25 indexes. Call this after a new extraction run."""
        logger.info("Refreshing vector store and retriever...")
        self.vectorstore = load_vector_store(self.embeddings)
        if self.vectorstore:
            self.retriever = HybridRetriever(self.db, self.vectorstore)
        else:
            logger.warning("Vector store still not found after refresh.")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline for the given query:
        Retrieves context, invokes the LLM, verifies citations, and runs NLI validation.
        """
        t_total_start = time.perf_counter()
        latencies = {}
        
        # Guard clause: retriever not initialized
        if not self.retriever:
            self.refresh_retriever()
            if not self.retriever:
                return {
                    "question": query,
                    "answer": "System is not initialized. Please load repository data and build the vector index first.",
                    "original_answer": "",
                    "retrieved_docs": [],
                    "citations": {"commits": {}, "prs": {}, "issues": {}},
                    "confidence_score": 0.0,
                    "latencies": {"total_response_latency": 0.0}
                }
                
        # 1. Retrieve context
        logger.info(f"Retrieving context for query: '{query}'")
        retrieved_docs, retrieval_metrics = self.retriever.retrieve(query)
        latencies.update(retrieval_metrics)
        
        # Format context for prompt
        context_str = ""
        if retrieved_docs:
            context_str = "\n\n".join([
                f"Source [{doc.metadata.get('type').upper()} #{doc.metadata.get('id')}]:\n{doc.page_content}"
                for doc in retrieved_docs
            ])
        else:
            context_str = "No relevant context found."
            
        # 2. LLM Generation
        t_llm_start = time.perf_counter()
        provider = settings.llm_provider.lower()
        logger.info(f"Generating answer via LLM ({provider})...")
            
        try:
            prompt_val = QA_PROMPT.format_prompt(context=context_str, question=query)
            response = self.llm.invoke(prompt_val.to_messages())
            raw_answer = response.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            raw_answer = f"An error occurred while calling the language model ({provider})."
        latencies["llm_latency"] = time.perf_counter() - t_llm_start
        
        # 3. Citation Verification (Hallucination Guard)
        logger.info("Verifying citations in generated answer...")
        cleaned_answer, citation_results, verify_metrics = self.guard.verify_citations(raw_answer, retrieved_docs)
        latencies.update(verify_metrics)
        
        # 4. NLI validation
        logger.info("Computing NLI entailment confidence score...")
        cleaned_answer, nli_score, nli_metrics = self.guard.calculate_nli_entailment(cleaned_answer, retrieved_docs)
        latencies.update(nli_metrics)
        
        # Convert verified citation tags to markdown links
        cleaned_answer = self.guard.format_citations_as_markdown(cleaned_answer)
        
        # Record total response time
        latencies["total_response_latency"] = time.perf_counter() - t_total_start
        
        logger.info(f"Pipeline executed successfully in {latencies['total_response_latency']:.4f}s")
        
        return {
            "question": query,
            "answer": cleaned_answer,
            "original_answer": raw_answer,
            "retrieved_docs": retrieved_docs,
            "citations": citation_results,
            "confidence_score": nli_score,
            "latencies": latencies
        }
