import logging
from app.database import DatabaseManager
from app.rag_pipeline import PatchContextRAG

logging.basicConfig(level=logging.INFO)

db = DatabaseManager('data/metadata.db')
rag = PatchContextRAG(db)

query = "What labels does Issue #458 have?"
print(f"\nRunning RAG pipeline for query: '{query}'")
result = rag.run(query)
print("\n--- RESULTS ---")
print("Question:", result["question"])
print("Original Answer (Raw LLM):", result["original_answer"])
print("Final Answer:", result["answer"])
print("Confidence Score:", result["confidence_score"])
print("Citations:", result["citations"])
print("Latencies:", result["latencies"])
