import time
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from app.database import DatabaseManager
from app.parser import DocumentParser

logger = logging.getLogger("PatchContext.Retriever")

class HybridRetriever:
    """Performs hybrid retrieval using FAISS Vector Search (MMR) and BM25, 
    expands context using the SQLite relationship graph, and re-ranks with a Cross-Encoder.
    """
    
    def __init__(self, db: DatabaseManager, vectorstore: FAISS):
        self.db = db
        self.vectorstore = vectorstore
        self.parser = DocumentParser()
        
        # Initialize LangChain MMR vector retriever
        if self.vectorstore:
            self.vector_retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.7}
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
            # Configure BM25 retrieve count
            self.bm25_retriever.k = 15
            logger.info(f"BM25 index built with {len(chunked_docs)} chunks.")
        else:
            logger.warning("No indexed documents found in database. BM25 retriever initialized empty.")
            self.bm25_retriever = None

    def retrieve(self, query: str, k_final: int = 5) -> Tuple[List[Document], Dict[str, float]]:
        """
        Executes the full hybrid retrieval, graph expansion, and re-ranking pipeline.
        Returns:
            Tuple of:
                - List of top-k re-ranked Documents
                - Dictionary containing performance latency metrics (in seconds)
        """
        metrics = {}
        t_total_start = time.perf_counter()
        
        # 1. Vector Search (MMR)
        t_start = time.perf_counter()
        vector_docs = []
        if self.vector_retriever:
            try:
                # We retrieve 15 documents via MMR search first to pass to the reranker
                vector_docs = self.vector_retriever.invoke(query)
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
        metrics["vector_latency"] = time.perf_counter() - t_start
        
        # 2. BM25 Search
        t_start = time.perf_counter()
        bm25_docs = []
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")
        metrics["bm25_latency"] = time.perf_counter() - t_start
        
        # 3. Merge & Deduplicate candidates
        candidates_map: Dict[str, Document] = {}
        
        def add_doc_to_candidates(doc: Document):
            # Form unique key based on type + ID + content chunk hash or content prefix
            m = doc.metadata
            key = f"{m.get('type')}_{m.get('id')}_{hash(doc.page_content[:150])}"
            if key not in candidates_map:
                candidates_map[key] = doc
                
        for doc in vector_docs:
            add_doc_to_candidates(doc)
        for doc in bm25_docs:
            add_doc_to_candidates(doc)
            
        initial_candidates = list(candidates_map.values())
        logger.debug(f"Retrieved {len(initial_candidates)} candidate chunks from hybrid search.")
        
        # 4. Graph-Aware Context Expansion
        t_start = time.perf_counter()
        expanded_docs = self._expand_context_graph(initial_candidates)
        metrics["graph_expansion_latency"] = time.perf_counter() - t_start
        
        # Merge initial and graph-expanded docs
        for doc in expanded_docs:
            add_doc_to_candidates(doc)
            
        all_candidates = list(candidates_map.values())
        logger.debug(f"Total candidates after graph expansion: {len(all_candidates)}")
        
        # 5. Cross-Encoder Re-ranking
        t_start = time.perf_counter()
        final_docs = []
        if all_candidates:
            # Pair query with each candidate document content
            pairs = [[query, doc.page_content] for doc in all_candidates]
            scores = self.reranker.predict(pairs)
            
            # Sort by score descending
            scored_candidates = sorted(
                zip(all_candidates, scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Save cross-encoder score in document metadata
            for doc, score in scored_candidates:
                doc.metadata["rerank_score"] = float(score)
                
            # Take top k_final
            final_docs = [doc for doc, _ in scored_candidates[:k_final]]
        metrics["rerank_latency"] = time.perf_counter() - t_start
        
        metrics["total_retrieval_latency"] = time.perf_counter() - t_total_start
        
        logger.info(
            f"Hybrid retrieval finished: {len(final_docs)} chunks returned. "
            f"Total retrieval latency: {metrics['total_retrieval_latency']:.4f}s"
        )
        return final_docs, metrics

    def _expand_context_graph(self, docs: List[Document], max_expansion: int = 5) -> List[Document]:
        """Queries the SQLite relationships table to fetch related nodes for the candidate documents."""
        expanded_documents = []
        added_keys = set()
        
        for doc in docs:
            if len(expanded_documents) >= max_expansion:
                break
                
            m = doc.metadata
            item_type = m.get("type")
            item_id = m.get("id")
            
            if not item_type or not item_id:
                continue
                
            # Get related items from database relationships
            related = self.db.get_related_items(item_type, item_id)
            for rel_type, rel_id, link in related:
                if len(expanded_documents) >= max_expansion:
                    break
                    
                key = f"{rel_type}_{rel_id}"
                if key in added_keys:
                    continue
                added_keys.add(key)
                
                # Retrieve record from SQLite
                row = self.db.get_item(rel_type, rel_id)
                if row:
                    logger.debug(f"Graph expansion: linked {item_type} #{item_id} -> {rel_type} #{rel_id} ({link})")
                    # Parse row to Document
                    related_doc = None
                    if rel_type == "commit" or rel_type == "commits":
                        related_doc = self.parser.parse_commit(row)
                    elif rel_type == "pr" or rel_type == "prs":
                        related_doc = self.parser.parse_pr(row)
                    elif rel_type == "issue" or rel_type == "issues":
                        related_doc = self.parser.parse_issue(row)
                        
                    if related_doc:
                        # Append the related document chunked or as a whole.
                        # Since we want it as context, let's chunk it so it fits nicely
                        chunks = self.parser.splitter.split_documents([related_doc])
                        for chunk in chunks:
                            chunk.metadata["graph_expanded"] = True
                            chunk.metadata["expanded_from"] = f"{item_type}:{item_id}"
                            expanded_documents.append(chunk)
                            
        return expanded_documents
