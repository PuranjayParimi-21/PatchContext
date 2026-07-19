import logging
from app.database import DatabaseManager
from app.retriever import HybridRetriever
from app.vector_store import load_vector_store
from app.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO)

db = DatabaseManager('data/metadata.db')
embeddings = get_embeddings()
vectorstore = load_vector_store(embeddings)

retriever = HybridRetriever(db, vectorstore)
try:
    docs, metrics = retriever.retrieve("Explain commit 6a274d1.")
    print("SUCCESS!")
    print("Docs returned:", len(docs))
    print("Metrics:", metrics)
except Exception as e:
    import traceback
    traceback.print_exc()
