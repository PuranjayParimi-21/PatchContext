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
        self.tokenizer = None
        self.nli_model = None
        self._init_nli_model()
        
    def _init_nli_model(self) -> None:
        """Loads the tokenizer and model for BART MNLI if NLI verification is enabled."""
        if not settings.enable_nli_guard:
            logger.info("NLI Hallucination Guard is disabled in settings. Skipping model load.")
            return
            
        try:
            logger.info(f"Loading NLI model and tokenizer: {settings.nli_model_name}...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(settings.nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model_name)
            
            # Use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.nli_model.to(self.device)
            logger.info(f"NLI model loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}. NLI validation will be bypassed.", exc_info=True)
            self.tokenizer = None
            self.nli_model = None

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
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Uses BART MNLI sequence classification to calculate confidence score.
        For every claim/sentence carrying a citation:
        1. Identifies the citation references (PR, Commit, Issue).
        2. Retrieves the supporting documents from the retrieved context for those citations.
        3. Formulates a premise-hypothesis pair: premise=supporting context, hypothesis=sentence/claim.
        4. Runs sequence classification and extracts probability of the entailment label (index 2).
        5. If the score is below the configured NLI threshold, the sentence is stripped.
        """
        t_start = time.perf_counter()
        
        # Guard clause: bypass NLI if not initialized, if answer is empty, or if we already rejected it
        if not self.nli_model or not self.tokenizer or not answer or answer == "I couldn't find sufficient evidence.":
            return answer, 1.0, {"nli_eval_latency": time.perf_counter() - t_start}
            
        try:
            import torch
            
            # Split answer into sentences
            raw_sentences = re.split(r'(?<=[.!?])\s+', answer)
            sentences = [s.strip() for s in raw_sentences if s.strip()]
            
            valid_sentences = []
            scores = []
            threshold = settings.nli_entailment_threshold
            
            for sentence in sentences:
                # Check if this sentence contains any citations
                parsed = self.parse_citations(sentence)
                has_citations = any(parsed[category] for category in parsed)
                
                # If a sentence does not carry any citations, keep it as is (no NLI check needed)
                if not has_citations:
                    valid_sentences.append(sentence)
                    continue
                    
                # Identify supporting context for the citations in this sentence
                supporting_texts = []
                for doc in retrieved_docs:
                    meta = doc.metadata
                    dtype = str(meta.get("type", "")).lower()
                    did = str(meta.get("id", "")).lower()
                    
                    if dtype == "commit" and parsed["commits"]:
                        for sha in parsed["commits"]:
                            if sha.startswith(did) or did.startswith(sha):
                                supporting_texts.append(doc.page_content)
                                break
                    elif dtype == "pr" and parsed["prs"]:
                        number = str(meta.get("number", ""))
                        if number in parsed["prs"]:
                            supporting_texts.append(doc.page_content)
                    elif dtype == "issue" and parsed["issues"]:
                        number = str(meta.get("number", ""))
                        if number in parsed["issues"]:
                            supporting_texts.append(doc.page_content)
                            
                premise = "\n\n".join(supporting_texts).strip()
                
                # If there is no supporting context (e.g. citations not found in retrieved docs),
                # the claim is automatically unsupported.
                if not premise:
                    logger.info(f"NLI Guard: Removing sentence (no retrieved context matches citation): '{sentence}'")
                    continue
                    
                # Limit premise length to prevent model token overflow
                premise = premise[:4000]
                
                # Tokenize premise and hypothesis pair
                inputs = self.tokenizer(
                    premise, 
                    sentence, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1024
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.nli_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                    # Label index 2 corresponds to ENTAILMENT
                    entailment_score = float(probs[2].item())
                    
                scores.append(entailment_score)
                
                if entailment_score >= threshold:
                    valid_sentences.append(sentence)
                else:
                    logger.info(f"NLI Guard: Removing unsupported sentence: '{sentence}' (score: {entailment_score:.4f})")
                    
            if not valid_sentences:
                filtered_answer = "I couldn't find sufficient evidence."
                final_score = 0.0
            else:
                filtered_answer = " ".join(valid_sentences)
                final_score = sum(scores) / len(scores) if scores else 1.0
                
            logger.info(f"NLI Verification completed. Score: {final_score:.4f}")
            return filtered_answer, final_score, {"nli_eval_latency": time.perf_counter() - t_start}
            
        except Exception as e:
            logger.error(f"Error running NLI validation: {e}. Bypassing NLI score.")
            return answer, 1.0, {"nli_eval_latency": time.perf_counter() - t_start}

    def format_citations_as_markdown(self, text: str) -> str:
        """Converts [Commit <sha>], [PR <n>], [Issue <n>] tags into markdown links."""
        repo = settings.github_repository
        
        def replace_commit(match):
            sha = match.group(1)
            return f"[Commit {sha[:7]}](https://github.com/{repo}/commit/{sha})"
            
        def replace_pr(match):
            pr_num = match.group(1)
            return f"[PR #{pr_num}](https://github.com/{repo}/pull/{pr_num})"
            
        def replace_issue(match):
            issue_num = match.group(1)
            return f"[Issue #{issue_num}](https://github.com/{repo}/issues/{issue_num})"
            
        text = re.sub(r'\[Commit\s+([a-f0-9]+)\]', replace_commit, text, flags=re.IGNORECASE)
        text = re.sub(r'\[PR\s+(\d+)\]', replace_pr, text, flags=re.IGNORECASE)
        text = re.sub(r'\[Issue\s+(\d+)\]', replace_issue, text, flags=re.IGNORECASE)
        return text
