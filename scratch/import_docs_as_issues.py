import os
import sqlite3
import json
import logging
from app.config import settings
from app.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ImportDocs")

def main():
    db = DatabaseManager(settings.database_path)
    docs_dir = os.path.join("data", "fastapi_repo", "docs", "en", "docs")
    
    if not os.path.exists(docs_dir):
        logger.error(f"Docs directory not found at: {docs_dir}")
        return
        
    doc_index = 200000
    conn = db._get_connection()
    cursor = conn.cursor()
    
    # Traverse docs_dir recursively and load all .md files
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, docs_dir)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    # Title will represent the doc file path
                    title = f"Doc: {rel_path}"
                    body = f"Source Document Path: docs/en/docs/{rel_path}\n\nContent:\n{content}"
                    
                    # Insert as a pseudo-issue
                    # Avoid duplicate titles if re-run
                    cursor.execute("SELECT number FROM issues WHERE title = ?", (title,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        num = existing[0]
                        cursor.execute("""
                            UPDATE issues SET body = ?, is_indexed = 0 WHERE number = ?
                        """, (body, num))
                        logger.info(f"Updated doc: {rel_path} as Issue #{num}")
                    else:
                        num = doc_index
                        cursor.execute("""
                            INSERT INTO issues (number, title, body, comments, labels, created_at, is_indexed)
                            VALUES (?, ?, ?, ?, ?, ?, 0)
                        """, (num, title, body, json.dumps([]), json.dumps(["documentation"]), ""))
                        logger.info(f"Imported doc: {rel_path} as Issue #{num}")
                        doc_index += 1
                        
                except Exception as e:
                    logger.error(f"Error importing {file_path}: {e}")
                    
    conn.commit()
    conn.close()
    logger.info("Documentation import completed successfully.")

if __name__ == "__main__":
    main()
