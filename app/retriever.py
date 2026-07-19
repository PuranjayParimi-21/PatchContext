import time
import logging
import re
import hashlib
import os
import git
import requests
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from app.database import DatabaseManager
from app.parser import DocumentParser
from app.config import settings

logger = logging.getLogger("PatchContext.Retriever")

class HybridRetriever:
    """Performs hybrid retrieval where LangChain's vector search via MMR (Maximal Marginal Relevance)
    acts as the primary retrieval mechanism for semantic diversity. Other stages (BM25 search, 
    SQLite relationship graph expansion, and Cross-Encoder re-ranking) serve only as supplementary 
    stages to enhance and filter the MMR-retrieved candidates.
    """
    
    def __init__(self, db: DatabaseManager, vectorstore: FAISS):
        self.db = db
        self.vectorstore = vectorstore
        self.parser = DocumentParser()
        
        # Initialize LangChain MMR vector retriever
        if self.vectorstore:
            self.vector_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 25, "fetch_k": 60, "lambda_mult": 0.5}
            )
        else:
            self.vector_retriever = None
            
        # Load Cross-Encoder model
        logger.info("Loading Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-Encoder model loaded successfully.")
        
        # Initialize BM25 retriever
        self.bm25_retriever = None
        self._build_bm25_retriever()
        
    def _build_bm25_retriever(self) -> None:
        """Loads all documents from the database and builds the BM25 index."""
        logger.info("Rebuilding BM25 retriever index from database...")
        docs: List[Document] = []
        
        # Pull all indexed items from DB to build BM25 retriever
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch commits
            cursor.execute("SELECT * FROM commits WHERE is_indexed = 1")
            for row in cursor.fetchall():
                docs.append(self.parser.parse_commit(dict(row)))
                
            # Fetch PRs
            cursor.execute("SELECT * FROM prs WHERE is_indexed = 1")
            for row in cursor.fetchall():
                docs.append(self.parser.parse_pr(dict(row)))
                
            # Fetch Issues
            cursor.execute("SELECT * FROM issues WHERE is_indexed = 1")
            for row in cursor.fetchall():
                docs.append(self.parser.parse_issue(dict(row)))
                
        if docs:
            # We chunk the documents using our parser's splitter so BM25 operates on the same chunks
            chunked_docs = self.parser.splitter.split_documents(docs)
            self.bm25_retriever = BM25Retriever.from_documents(chunked_docs)
            # Configure BM25 retrieve count (k = 25)
            self.bm25_retriever.k = 25
            logger.info(f"BM25 index built with {len(chunked_docs)} chunks.")
        else:
            logger.warning("No indexed documents found in database. BM25 retriever initialized empty.")
            self.bm25_retriever = None

    def _detect_identifier(self, query: str) -> Optional[Tuple[str, str]]:
        """Detects if the query references a specific PR, Issue, or Commit SHA using regular expressions.
        Returns:
            Tuple of (type, id) if detected, else None.
        """
        # 1. Pull Request patterns (e.g. PR #145, Pull Request 145, PR145)
        pr_match = re.search(r"(?i)\b(?:pull\s*request|pr)\b\s*#?(\d+)", query)
        if pr_match:
            pr_id = pr_match.group(1)
            logger.info(f"Identifier detection: detected Pull Request #{pr_id} in query.")
            return "pr", pr_id
            
        # 2. Issue patterns (e.g. Issue #458, Issue 458)
        issue_match = re.search(r"(?i)\bissue\b\s*#?(\d+)", query)
        if issue_match:
            issue_id = issue_match.group(1)
            logger.info(f"Identifier detection: detected Issue #{issue_id} in query.")
            return "issue", issue_id
            
        # 3. Commit SHA patterns
        # Look for "commit <sha>" prefix
        commit_prefix_match = re.search(r"(?i)\bcommit\b\s+([0-9a-f]{7,40})\b", query)
        if commit_prefix_match:
            sha = commit_prefix_match.group(1).lower()
            logger.info(f"Identifier detection: detected Commit prefix with SHA '{sha}' in query.")
            return "commit", sha

        # Look for raw commit SHA hashes (7-40 hex characters)
        hex_matches = re.findall(r"\b([0-9a-f]{7,40})\b", query, re.IGNORECASE)
        for sha in hex_matches:
            sha_lower = sha.lower()
            # If purely numeric and short, ignore to avoid conflict with issue/PR numbers
            if sha_lower.isdigit() and len(sha_lower) < 10:
                continue
            # Validate if it exists as a commit in the database (supports prefix/abbreviation checks)
            if self._exists_commit_sha(sha_lower):
                logger.info(f"Identifier detection: detected valid Commit SHA '{sha_lower}' in query.")
                return "commit", sha_lower
                
        return None

    def _exists_commit_sha(self, sha: str) -> bool:
        """Helper to check if a full or abbreviated SHA exists in the SQLite database."""
        if len(sha) == 40:
            return self.db.exists_in_db("commit", sha)
        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM commits WHERE sha LIKE ? LIMIT 1", (f"{sha.lower()}%",))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking exists for SHA prefix '{sha}': {e}")
            return False

    def _exact_lookup(self, item_type: str, item_id: str) -> List[Document]:
        """Performs a direct lookup in SQLite for the specified item type and id."""
        logger.info(f"SQLite lookup: fetching {item_type} '{item_id}'...")
        row = None
        
        # Support abbreviated commit SHA lookups
        if item_type == "commit" and len(item_id) < 40:
            try:
                with self.db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM commits WHERE sha LIKE ? LIMIT 1", (f"{item_id.lower()}%",))
                    db_row = cursor.fetchone()
                    if db_row:
                        row = dict(db_row)
            except Exception as e:
                logger.error(f"Error retrieving commit by SHA prefix '{item_id}': {e}")
        else:
            row = self.db.get_item(item_type, item_id)
            
        # Self-healing dynamic imports for unindexed entities
        if not row:
            if item_type == "commit":
                logger.info(f"Commit '{item_id}' not found in SQLite DB. Attempting dynamic import from local git repo...")
                try:
                    git_repo_path = settings.local_repo_path
                    git_dir = os.path.join(git_repo_path, ".git")
                    
                    # On-demand cloning if git repo is missing or invalid on deployment server
                    if not os.path.exists(git_repo_path) or not os.path.exists(git_dir):
                        logger.info(f"Git repository missing at {git_repo_path}. Cloning dynamically...")
                        if os.path.exists(git_repo_path):
                            import shutil
                            try:
                                shutil.rmtree(git_repo_path)
                            except Exception:
                                pass
                        repo_url = "https://github.com/" + settings.github_repository
                        git.Repo.clone_from(repo_url, git_repo_path)
                        logger.info("Dynamic repository clone completed.")
                        
                    repo = git.Repo(git_repo_path)
                    commit = repo.commit(item_id)
                    sha = commit.hexsha
                    
                    # Compute files and diff
                    changed_files = []
                    diff_text = ""
                    if commit.parents:
                        diffs = commit.parents[0].diff(commit, create_patch=True)
                    else:
                        diffs = commit.diff(None, create_patch=True)
                        
                    for d in diffs:
                        a_path = d.a_path or ""
                        b_path = d.b_path or ""
                        path = b_path if b_path else a_path
                        if path:
                            changed_files.append(path)
                        if d.diff:
                            try:
                                patch = d.diff.decode('utf-8', errors='ignore')
                                if len(diff_text) + len(patch) < 6000:
                                    diff_text += f"\nFile: {path}\n{patch}"
                            except Exception:
                                pass
                                
                    # Insert into SQLite database so it exists
                    self.db.insert_commit(
                        sha=sha,
                        author=commit.author.name,
                        date=commit.committed_datetime.isoformat(),
                        message=commit.message or "",
                        changed_files=changed_files,
                        diff=diff_text
                    )
                    logger.info(f"Successfully dynamically indexed commit {sha[:7]} into SQLite database.")
                    row = self.db.get_item("commit", sha)
                except Exception as e:
                    logger.error(f"Failed to dynamically import commit '{item_id}' from git: {e}")
                    
            elif item_type == "pr" and settings.github_token and settings.github_token != "missing-api-key":
                logger.info(f"PR '{item_id}' not found in SQLite DB. Attempting dynamic import from GitHub API...")
                try:
                    repo_slug = settings.github_repository
                    url = f"https://api.github.com/repos/{repo_slug}/pulls/{item_id}"
                    headers = {"Authorization": f"Bearer {settings.github_token}"}
                    resp = requests.get(url, headers=headers)
                    if resp.status_code == 200:
                        pr_data = resp.json()
                        self.db.insert_pr(
                            number=int(item_id),
                            title=pr_data.get("title", ""),
                            body=pr_data.get("body", ""),
                            comments=[],
                            review_comments=[],
                            merged_commit_sha=pr_data.get("merge_commit_sha"),
                            created_at=pr_data.get("created_at", "")
                        )
                        merged_sha = pr_data.get("merge_commit_sha")
                        if merged_sha:
                            self.db.insert_relationship("pr", item_id, "commit", merged_sha, "merges")
                            
                        logger.info(f"Successfully dynamically indexed PR #{item_id} from GitHub API.")
                        row = self.db.get_item("pr", item_id)
                except Exception as e:
                    logger.error(f"Failed to dynamically import PR #{item_id}: {e}")
                    
            elif item_type == "issue" and settings.github_token and settings.github_token != "missing-api-key":
                logger.info(f"Issue '{item_id}' not found in SQLite DB. Attempting dynamic import from GitHub API...")
                try:
                    repo_slug = settings.github_repository
                    url = f"https://api.github.com/repos/{repo_slug}/issues/{item_id}"
                    headers = {"Authorization": f"Bearer {settings.github_token}"}
                    resp = requests.get(url, headers=headers)
                    if resp.status_code == 200:
                        issue_data = resp.json()
                        self.db.insert_issue(
                            number=int(item_id),
                            title=issue_data.get("title", ""),
                            body=issue_data.get("body", ""),
                            comments=[],
                            labels=[l.get("name") for l in issue_data.get("labels", [])],
                            created_at=issue_data.get("created_at", "")
                        )
                        logger.info(f"Successfully dynamically indexed Issue #{item_id} from GitHub API.")
                        row = self.db.get_item("issue", item_id)
                except Exception as e:
                    logger.error(f"Failed to dynamically import Issue #{item_id}: {e}")
                    
        if not row:
            logger.warning(f"SQLite lookup: no database record found for {item_type} '{item_id}'.")
            return []
            
        doc = None
        if item_type == "commit":
            doc = self.parser.parse_commit(row)
        elif item_type == "pr":
            doc = self.parser.parse_pr(row)
        elif item_type == "issue":
            doc = self.parser.parse_issue(row)
            
        if doc:
            chunks = self.parser.splitter.split_documents([doc])
            logger.info(f"SQLite lookup: successfully retrieved {item_type} '{item_id}'. Split into {len(chunks)} chunks.")
            return chunks
            
        return []

    def _vector_search(self, query: str) -> List[Document]:
        """Performs FAISS semantic search using Max Marginal Relevance (MMR)."""
        if not self.vector_retriever:
            logger.warning("Vector search requested but vector store is not initialized.")
            return []
            
        logger.info("FAISS vector search: executing MMR search...")
        try:
            docs = self.vector_retriever.invoke(query)
            logger.info(f"FAISS count: retrieved {len(docs)} documents.")
            return docs
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def _bm25_search(self, query: str) -> List[Document]:
        """Performs lexical keyword search using BM25."""
        if not self.bm25_retriever:
            logger.warning("BM25 search requested but retriever is not initialized.")
            return []
            
        logger.info("BM25 search: executing query...")
        try:
            docs = self.bm25_retriever.invoke(query)
            logger.info(f"BM25 count: retrieved {len(docs)} documents.")
            return docs
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _merge_candidates(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """Merges vector search and BM25 candidates, deduplicating using MD5 content hash."""
        candidates_map: Dict[str, Document] = {}
        
        def add_doc(doc: Document):
            m = doc.metadata
            doc_type = m.get("type", "unknown")
            doc_id = str(m.get("id", ""))
            
            # Compute stable process-independent content hash
            content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            key = f"{doc_type}_{doc_id}_{content_hash}"
            
            if key not in candidates_map:
                candidates_map[key] = doc
                
        for doc in vector_docs:
            add_doc(doc)
        for doc in bm25_docs:
            add_doc(doc)
            
        merged = list(candidates_map.values())
        logger.info(f"Merged count: combined into {len(merged)} unique document chunks.")
        return merged

    def _expand_context_graph(self, docs: List[Document], max_expansion: int = 10) -> List[Document]:
        """Queries the SQLite relationships table to fetch related nodes for the candidate documents.
        Enforces a strict expansion limit of up to `max_expansion` related nodes.
        """
        if not docs:
            return []
            
        expanded_documents = []
        
        # Track items that are already in the candidates list to avoid expanding or re-fetching them
        existing_items = set()
        for doc in docs:
            m = doc.metadata
            item_type = m.get("type")
            item_id = m.get("id")
            if item_type and item_id:
                # Normalize types to singular
                t = item_type.lower()
                if t.endswith("s"):
                    t = t[:-1]
                existing_items.add((t, str(item_id)))
                
        expanded_keys = set()
        expansion_count = 0
        
        logger.info(f"Graph expansion: checking relationships for {len(docs)} candidates...")
        
        for doc in docs:
            if expansion_count >= max_expansion:
                break
                
            m = doc.metadata
            item_type = m.get("type")
            item_id = m.get("id")
            
            if not item_type or not item_id:
                continue
                
            # Get related items from database relationships
            related = self.db.get_related_items(item_type, item_id)
            for rel_type, rel_id, link in related:
                if expansion_count >= max_expansion:
                    break
                    
                # Normalize types to singular
                r_type = rel_type.lower()
                if r_type.endswith("s"):
                    r_type = r_type[:-1]
                r_id = str(rel_id)
                
                # Skip if already in candidates or already expanded
                if (r_type, r_id) in existing_items or (r_type, r_id) in expanded_keys:
                    continue
                    
                expanded_keys.add((r_type, r_id))
                
                # Retrieve record from SQLite
                row = self.db.get_item(r_type, r_id)
                if row:
                    logger.info(f"Graph expansion: linked {item_type} #{item_id} -> {r_type} #{r_id} ({link})")
                    related_doc = None
                    if r_type == "commit":
                        related_doc = self.parser.parse_commit(row)
                    elif r_type == "pr":
                        related_doc = self.parser.parse_pr(row)
                    elif r_type == "issue":
                        related_doc = self.parser.parse_issue(row)
                        
                    if related_doc:
                        expansion_count += 1
                        # Chunk the related document
                        chunks = self.parser.splitter.split_documents([related_doc])
                        for chunk in chunks:
                            chunk.metadata["graph_expanded"] = True
                            chunk.metadata["expanded_from"] = f"{item_type}:{item_id}"
                            expanded_documents.append(chunk)
                            
        logger.info(f"Expanded count: added {len(expanded_documents)} chunks from {expansion_count} related nodes.")
        return expanded_documents

    def _rerank(self, query: str, candidates: List[Document], k_final: int = 10) -> List[Document]:
        """Reranks candidates using the Cross-Encoder model and selects the top k_final items."""
        if not candidates:
            logger.info("Reranking: no candidate documents to rerank.")
            return []
            
        logger.info(f"Reranking: predicting relevance scores for {len(candidates)} candidates...")
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Pair documents with scores and sort descending
        scored_candidates = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info("CrossEncoder scores:")
        for doc, score in scored_candidates:
            doc.metadata["rerank_score"] = float(score)
            snippet = doc.page_content[:60].replace('\n', ' ')
            logger.info(f"  - Score {score:.4f} | {doc.metadata.get('type')} #{doc.metadata.get('id')} | Snippet: {snippet}...")
            
        # Select top k_final
        final_docs = [doc for doc, _ in scored_candidates[:k_final]]
        
        logger.info(f"Final selected documents (top {len(final_docs)}):")
        for i, doc in enumerate(final_docs):
            logger.info(f"  [{i+1}] {doc.metadata.get('type')} #{doc.metadata.get('id')} (score: {doc.metadata.get('rerank_score'):.4f})")
            
        return final_docs

    def retrieve(self, query: str, k_final: int = 10) -> Tuple[List[Document], Dict[str, float]]:
        """
        Executes the full retrieval, routing, graph expansion, and re-ranking pipeline.
        Returns:
            Tuple of:
                - List of top-k re-ranked Documents
                - Dictionary containing performance latency metrics (in seconds)
        """
        metrics = {}
        t_total_start = time.perf_counter()
        
        # 1. Query routing / exact identifier detection
        t_start = time.perf_counter()
        identifier = self._detect_identifier(query)
        metrics["identifier_detect_latency"] = time.perf_counter() - t_start
        
        if identifier:
            item_type, item_id = identifier
            logger.info(f"Query routing: matched exact entity identifier '{item_type}:{item_id}'. Bypassing semantic search.")
            
            # Fast-track path: SQLite lookup -> Relationship Expansion -> CrossEncoder
            t_start = time.perf_counter()
            exact_docs = self._exact_lookup(item_type, item_id)
            metrics["exact_lookup_latency"] = time.perf_counter() - t_start
            
            # Graph relationship expansion (up to 10 nodes)
            t_start = time.perf_counter()
            expanded_docs = self._expand_context_graph(exact_docs, max_expansion=10)
            metrics["graph_expansion_latency"] = time.perf_counter() - t_start
            
            # Merge and deduplicate candidates
            candidates = self._merge_candidates(exact_docs, expanded_docs)
            
            # Rerank
            t_start = time.perf_counter()
            final_docs = self._rerank(query, candidates, k_final=k_final)
            metrics["rerank_latency"] = time.perf_counter() - t_start
            
            # Zero out unused search latencies to maintain downstream compatibility
            metrics["vector_latency"] = 0.0
            metrics["bm25_latency"] = 0.0
            metrics["total_retrieval_latency"] = time.perf_counter() - t_total_start
            
            logger.info(
                f"Exact lookup finished: returned {len(final_docs)} chunks. "
                f"Total latency: {metrics['total_retrieval_latency']:.4f}s"
            )
            return final_docs, metrics
            
        else:
            logger.info("Query routing: no exact identifier matched. Executing hybrid retrieval pipeline.")
            
            # Hybrid search path: FAISS (MMR) + BM25 -> Merge -> Graph Expansion -> CrossEncoder
            
            # Vector search (MMR)
            t_start = time.perf_counter()
            vector_docs = self._vector_search(query)
            metrics["vector_latency"] = time.perf_counter() - t_start
            
            # BM25 Search
            t_start = time.perf_counter()
            bm25_docs = self._bm25_search(query)
            metrics["bm25_latency"] = time.perf_counter() - t_start
            
            # Merge & Deduplicate initial search results
            merged_search_docs = self._merge_candidates(vector_docs, bm25_docs)
            
            # Graph relationship expansion (up to 10 nodes)
            t_start = time.perf_counter()
            expanded_docs = self._expand_context_graph(merged_search_docs, max_expansion=10)
            metrics["graph_expansion_latency"] = time.perf_counter() - t_start
            
            # Merge expanded nodes back into primary candidate pool
            all_candidates = self._merge_candidates(merged_search_docs, expanded_docs)
            
            # Cross-Encoder Reranking
            t_start = time.perf_counter()
            final_docs = self._rerank(query, all_candidates, k_final=k_final)
            metrics["rerank_latency"] = time.perf_counter() - t_start
            
            metrics["total_retrieval_latency"] = time.perf_counter() - t_total_start
            
            logger.info(
                f"Hybrid retrieval finished: returned {len(final_docs)} chunks. "
                f"Total latency: {metrics['total_retrieval_latency']:.4f}s"
            )
            return final_docs, metrics
