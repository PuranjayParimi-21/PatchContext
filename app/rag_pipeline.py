import time
import logging
from typing import Dict, Any, List
# pyrefly: ignore [missing-import]
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
                temperature=0.3,
                max_tokens=512,
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
        if not retrieved_docs:
            logger.info("No relevant context retrieved. Returning default insufficient evidence message.")
            latencies["total_response_latency"] = time.perf_counter() - t_total_start
            return {
                "question": query,
                "answer": "I couldn't find sufficient evidence.",
                "original_answer": "No context retrieved.",
                "retrieved_docs": [],
                "citations": {"commits": {}, "prs": {}, "issues": {}},
                "confidence_score": 0.0,
                "latencies": latencies
            }
            
        # Truncate context to prevent exceeding free model token limits
        # Free models (3B-9B) have ~4K token context window; keep context small
        MAX_CONTEXT_CHARS = 4000
        context_parts = [
            f"Source [{doc.metadata.get('type').upper()} #{doc.metadata.get('id')}]:\n{doc.page_content[:800]}"
            for doc in retrieved_docs[:5]
        ]
        context_str = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]
            
        # 2. LLM Generation
        t_llm_start = time.perf_counter()
        provider = settings.llm_provider.lower()
        logger.info(f"Generating answer via LLM ({provider})...")
        
        llm_error = False
        try:
            prompt_val = QA_PROMPT.format_prompt(context=context_str, question=query)
            response = self.llm.invoke(prompt_val.to_messages())
            raw_answer = response.content
            # Treat empty model response as an error
            if not raw_answer or not raw_answer.strip():
                raise ValueError("Model returned an empty response.")
        except Exception as e:
            logger.error(f"Error generating answer with {self.llm.model_name}: {e}")
            llm_error = True
            if provider == "openrouter":
                # Valid free models available on OpenRouter as of 2025
                fallbacks = [
                    "meta-llama/llama-3.1-8b-instruct:free",
                    "meta-llama/llama-3.2-3b-instruct:free",
                    "microsoft/phi-3-mini-128k-instruct:free",
                    "google/gemma-2-9b-it:free",
                    "qwen/qwen-2-7b-instruct:free",
                ]
                success = False
                for fallback_model in fallbacks:
                    if fallback_model == self.llm.model_name:
                        continue
                    logger.info(f"Retrying with fallback model: {fallback_model}...")
                    self.llm.model_name = fallback_model
                    try:
                        # Brief pause to avoid rate limiting
                        time.sleep(1.5)
                        response = self.llm.invoke(prompt_val.to_messages())
                        raw_answer = response.content
                        # Also treat empty fallback response as failure
                        if not raw_answer or not raw_answer.strip():
                            raise ValueError(f"Fallback model {fallback_model} returned an empty response.")
                        success = True
                        llm_error = False
                        logger.info(f"Successfully generated answer using fallback model: {fallback_model}")
                        break
                    except Exception as fallback_err:
                        err_str = str(fallback_err)
                        if "429" in err_str or "rate limit" in err_str.lower():
                            logger.warning(f"Rate limited on {fallback_model}. Waiting 3s...")
                            time.sleep(3.0)
                        elif "model output" in err_str.lower() or "empty" in err_str.lower():
                            logger.warning(f"Model {fallback_model} returned empty output. Trying next fallback...")
                        logger.error(f"Fallback model {fallback_model} also failed: {fallback_err}")
                if not success:
                    raw_answer = f"An error occurred while calling the language model. Please try again in a moment."
            else:
                raw_answer = f"An error occurred while calling the language model. Please try again in a moment."
        latencies["llm_latency"] = time.perf_counter() - t_llm_start
        
        # Automatically attach missing citations from retrieved docs to raw_answer
        # IMPORTANT: Skip citation appending if LLM failed
        retrieved_citations = []
        if not llm_error:
            for doc in retrieved_docs:
                m = doc.metadata
                dtype = str(m.get("type", "")).lower()
                did = str(m.get("id", ""))
                if dtype == "commit":
                    retrieved_citations.append(f"[Commit {did}]")
                elif dtype == "pr":
                    retrieved_citations.append(f"[PR {did}]")
                elif dtype == "issue":
                    retrieved_citations.append(f"[Issue {did}]")
                    
            # Parse what citations are already in the raw_answer
            existing = self.guard.parse_citations(raw_answer)
            missing_cits = []
            for cit in retrieved_citations:
                temp_parsed = self.guard.parse_citations(cit)
                already_exists = False
                for sha in temp_parsed["commits"]:
                    if any(sha.startswith(s) or s.startswith(sha) for s in existing["commits"]):
                        already_exists = True
                for pr in temp_parsed["prs"]:
                    if pr in existing["prs"]:
                        already_exists = True
                for issue in temp_parsed["issues"]:
                    if issue in existing["issues"]:
                        already_exists = True
                if not already_exists:
                    missing_cits.append(cit)
                    
            if missing_cits:
                # Append missing citations at the end of raw_answer
                raw_answer += " (Citations: " + ", ".join(missing_cits) + ")"
        
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
