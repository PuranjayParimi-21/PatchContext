import re
import time
import logging
from typing import Dict, List, Any, Tuple, Set
from app.config import settings
from app.database import DatabaseManager
from langchain_core.documents import Document

logger = logging.getLogger("PatchContext.Verifier")

class HallucinationGuard:
    """Verifies citation existence in database and retrieved context, 
    and checks factual entailment using a BART NLI model.
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.nli_pipeline = None
        self._init_nli_model()
        
    def _init_nli_model(self) -> None:
        """Loads the BART MNLI model if NLI verification is enabled in configuration."""
        if not settings.enable_nli_guard:
            logger.info("NLI Hallucination Guard is disabled in settings. Skipping model load.")
            return
            
        try:
            logger.info(f"Loading NLI model: {settings.nli_model_name}...")
            # Import transformers only when NLI is enabled to speed up baseline loading
            from transformers import pipeline
            import torch
            
            device = 0 if torch.cuda.is_available() else -1
            self.nli_pipeline = pipeline(
                "zero-shot-classification",
                model=settings.nli_model_name,
                device=device
            )
            logger.info("NLI model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}. NLI validation will be bypassed.", exc_info=True)
            self.nli_pipeline = None

    def parse_citations(self, text: str) -> Dict[str, Set[str]]:
        """Parses strict citation patterns [Commit SHA], [PR Number], [Issue Number] from text."""
        citations = {
            "commits": set(),
            "prs": set(),
            "issues": set()
        }
        
        # Regex matching
        commit_matches = re.findall(r'\[Commit\s+([a-f0-9]+)\]', text, re.IGNORECASE)
        pr_matches = re.findall(r'\[PR\s+(\d+)\]', text, re.IGNORECASE)
        issue_matches = re.findall(r'\[Issue\s+(\d+)\]', text, re.IGNORECASE)
        
        for sha in commit_matches:
            citations["commits"].add(sha.lower())
        for pr in pr_matches:
            citations["prs"].add(pr)
        for issue in issue_matches:
            citations["issues"].add(issue)
            
        return citations

    def verify_citations(
        self, 
        answer: str, 
        retrieved_docs: List[Document]
    ) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
        """
        Validates all parsed citations in the generated answer:
        1. Checks if they exist in the SQLite database.
        2. Checks if they exist in the retrieved context.
        Removes any citations that fail either check, and measures latency.
        """
        t_start = time.perf_counter()
        parsed = self.parse_citations(answer)
        
        verification_results = {
            "commits": {},
            "prs": {},
            "issues": {}
        }
        
        unsupported_tokens: List[str] = []
        
        # 1. Verify Commits
        for sha in parsed["commits"]:
            exists_db = self.db.exists_in_db("commit", sha)
            
            # Check context match (matching full SHA or prefix)
            in_context = False
            for doc in retrieved_docs:
                meta = doc.metadata
                if meta.get("type") == "commit":
                    doc_sha = str(meta.get("sha", "")).lower()
                    if doc_sha.startswith(sha) or sha.startswith(doc_sha):
                        in_context = True
                        break
                        
            verified = exists_db and in_context
            verification_results["commits"][sha] = {
                "exists_db": exists_db,
                "in_context": in_context,
                "verified": verified
            }
            
            if not verified:
                # Find the exact matched casing in the answer to clean it
                match_pat = re.compile(rf'\[Commit\s+{sha}\]', re.IGNORECASE)
                unsupported_tokens.extend(match_pat.findall(answer))

        # 2. Verify PRs
        for pr_num in parsed["prs"]:
            exists_db = self.db.exists_in_db("pr", pr_num)
            
            in_context = False
            for doc in retrieved_docs:
                meta = doc.metadata
                if meta.get("type") == "pr" and str(meta.get("number")) == pr_num:
                    in_context = True
                    break
                    
            verified = exists_db and in_context
            verification_results["prs"][pr_num] = {
                "exists_db": exists_db,
                "in_context": in_context,
                "verified": verified
            }
            
            if not verified:
                match_pat = re.compile(rf'\[PR\s+{pr_num}\]', re.IGNORECASE)
                unsupported_tokens.extend(match_pat.findall(answer))

        # 3. Verify Issues
        for issue_num in parsed["issues"]:
            exists_db = self.db.exists_in_db("issue", issue_num)
            
            in_context = False
            for doc in retrieved_docs:
                meta = doc.metadata
                if meta.get("type") == "issue" and str(meta.get("number")) == issue_num:
                    in_context = True
                    break
                    
            verified = exists_db and in_context
            verification_results["issues"][issue_num] = {
                "exists_db": exists_db,
                "in_context": in_context,
                "verified": verified
            }
            
            if not verified:
                match_pat = re.compile(rf'\[Issue\s+{issue_num}\]', re.IGNORECASE)
                unsupported_tokens.extend(match_pat.findall(answer))

        # 4. Clean the answer by removing unsupported citation tokens
        cleaned_answer = answer
        for token in unsupported_tokens:
            cleaned_answer = cleaned_answer.replace(token, "")
            # Clean up double spaces or spaces before punctuation caused by deletion
            cleaned_answer = re.sub(r'\s+([,\.\?!])', r'\1', cleaned_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer)
            
        latency = {"citation_verification_latency": time.perf_counter() - t_start}
        return cleaned_answer.strip(), verification_results, latency

    def calculate_nli_entailment(
        self, 
        answer: str, 
        retrieved_docs: List[Document]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Uses BART zero-shot classification to calculate confidence score.
        Compares the premise (concatenated context) against the hypothesis (answer).
        """
        t_start = time.perf_counter()
        
        # Guard clause: bypass NLI if not initialized or if answer is empty
        if not self.nli_pipeline or not answer or answer == "I couldn't find sufficient evidence.":
            return 1.0, {"nli_eval_latency": time.perf_counter() - t_start}
            
        try:
            # Construct premise from top retrieval chunks
            premise = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # If premise is empty, confidence is zero
            if not premise.strip():
                return 0.0, {"nli_eval_latency": time.perf_counter() - t_start}
                
            # Limit premise length to prevent model token overflow
            premise = premise[:4000]
            
            result = self.nli_pipeline(
                sequences=answer,
                candidate_labels=["supported by context", "unsupported by context"],
                hypothesis_template=f"Based on this premise: {premise}. The statement: '{{}}' is true.",
            )
            
            # Find the score for 'supported by context' label
            supported_idx = result["labels"].index("supported by context")
            entailment_score = float(result["scores"][supported_idx])
            
            logger.info(f"NLI Verification score: {entailment_score:.4f}")
            return entailment_score, {"nli_eval_latency": time.perf_counter() - t_start}
            
        except Exception as e:
            logger.error(f"Error running NLI validation: {e}. Bypassing NLI score.")
            return 1.0, {"nli_eval_latency": time.perf_counter() - t_start}
