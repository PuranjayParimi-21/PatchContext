import sqlite3
import os
from app.embeddings import get_embeddings
from app.vector_store import load_vector_store

db_path = 'data/metadata.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM commits")
    commits_cnt = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM prs")
    prs_cnt = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM issues")
    issues_cnt = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM relationships")
    rel_cnt = cursor.fetchone()[0]
    
    print("--- Database Counts ---")
    print(f"Commits: {commits_cnt}")
    print(f"PRs: {prs_cnt}")
    print(f"Issues: {issues_cnt}")
    print(f"Relationships: {rel_cnt}")
else:
    print("DB does not exist")

vs_path = 'vectorstore'
if os.path.exists(vs_path):
    try:
        embeddings = get_embeddings()
        vs = load_vector_store(embeddings)
        if vs:
            print("--- Vector Store Status ---")
            print("Vector store loaded successfully.")
            # Print index size if accessible
            if hasattr(vs, 'index'):
                print("Index Size (number of vectors):", vs.index.ntotal)
    except Exception as e:
        print("Failed to load vector store:", e)
else:
    print("Vector store directory does not exist")
