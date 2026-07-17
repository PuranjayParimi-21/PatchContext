import os
import sys
import logging
from app.config import settings
from app.database import DatabaseManager
from app.parser import DocumentParser
from app.embeddings import get_embeddings
from app.vector_store import build_or_update_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RebuildIndex")

def main():
    logger.info("Initializing database manager...")
    db = DatabaseManager(settings.database_path)
    
    logger.info("Initializing document parser...")
    parser = DocumentParser()
    
    logger.info("Initializing embeddings model...")
    embeddings = get_embeddings()
    
    logger.info("Rebuilding FAISS index from SQLite records...")
    vectorstore = build_or_update_index(db, parser, embeddings)
    
    if vectorstore:
         logger.info("FAISS Vector Index successfully rebuilt and saved!")
    else:
         logger.error("Failed to build index or no records found.")

if __name__ == "__main__":
    main()
