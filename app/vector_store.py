import os
import logging
from typing import Optional, List, Set, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from app.config import settings
from app.database import DatabaseManager
from app.parser import DocumentParser

logger = logging.getLogger("PatchContext.VectorStore")

def load_vector_store(embeddings: Embeddings) -> Optional[FAISS]:
    """Loads the FAISS vector index from the configured path if it exists."""
    path = settings.vectorstore_path
    index_file = os.path.join(path, "index.faiss")
    if os.path.exists(index_file):
        logger.info(f"Loading existing FAISS vector store from {path}...")
        try:
            # allow_dangerous_deserialization is required for loading pickle-serialized FAISS files in LangChain
            store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            if store is not None:
                expected_dim = len(embeddings.embed_query("test"))
                actual_dim = store.index.d
                if actual_dim != expected_dim:
                    logger.warning(
                        f"FAISS index dimension mismatch! Embedding model expects {expected_dim} dimensions, "
                        f"but loaded index has {actual_dim} dimensions. Discarding stale index."
                    )
                    return None
            return store
        except Exception as e:
            logger.error(f"Error loading FAISS store: {e}", exc_info=True)
            return None
    logger.info("No existing FAISS index found.")
    return None

def build_or_update_index(
    db: DatabaseManager, 
    parser: DocumentParser, 
    embeddings: Embeddings
) -> Optional[FAISS]:
    """
    Checks SQLite for unindexed records, chunks them, embeds them,
    and adds them incrementally to the FAISS index. Marks records as indexed upon success.
    """
    logger.info("Checking for new documents to add to FAISS index...")
    
    # 1. Load existing FAISS index
    vectorstore = load_vector_store(embeddings)
    
    # If the vectorstore is missing, we reset the indexing status to force a full rebuild
    if vectorstore is None:
        logger.info("Vector store not found. Resetting database is_indexed flags to force full rebuild...")
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE commits SET is_indexed = 0")
            cursor.execute("UPDATE prs SET is_indexed = 0")
            cursor.execute("UPDATE issues SET is_indexed = 0")
            conn.commit()
            
    # 2. Get unindexed items parsed and chunked
    new_chunks = parser.process_unindexed_items(db)
    if not new_chunks:
        logger.info("FAISS Index is up to date. No new chunks added.")
        return vectorstore
        
    logger.info(f"Embedding and adding {len(new_chunks)} new chunks to FAISS index...")
    
    # 3. Add to index
    if vectorstore is not None:
        vectorstore.add_documents(new_chunks)
    else:
        logger.info("Initializing new FAISS index...")
        vectorstore = FAISS.from_documents(new_chunks, embeddings)
        
    # 4. Persist FAISS index
    os.makedirs(settings.vectorstore_path, exist_ok=True)
    vectorstore.save_local(settings.vectorstore_path)
    logger.info(f"FAISS index successfully saved to {settings.vectorstore_path}.")
    
    # 5. Mark source records as indexed in database
    indexed_items: Set[Tuple[str, str]] = set()
    for chunk in new_chunks:
        meta = chunk.metadata
        if "type" in meta and "id" in meta:
            # Map type representation (e.g. 'commit' -> 'commits', 'pr' -> 'prs', 'issue' -> 'issues')
            t = meta["type"]
            table_name = t + "s" if not t.endswith("s") else t
            indexed_items.add((table_name, meta["id"]))
            
    for table, item_id in indexed_items:
        db.mark_as_indexed(table, item_id)
        
    logger.info(f"Marked {len(indexed_items)} source items as indexed in SQLite database.")
    return vectorstore
