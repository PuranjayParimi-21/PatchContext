import logging
from app.database import DatabaseManager
from app.github_loader import GitHubLoader
from app.embeddings import get_embeddings
from app.parser import DocumentParser
from app.vector_store import build_or_update_index

logging.basicConfig(level=logging.INFO)

db = DatabaseManager('data/metadata.db')
loader = GitHubLoader(db)

print("\n=== STARTING BATCH REPOSITORY SYNC AND FAISS INDEXING ===")

print("\n[1/5] Extracting commits from local git history...")
try:
    loader.extract_commits(max_count=1000)
except Exception as e:
    print(f"Error extracting commits: {e}")

print("\n[2/5] Fetching PRs from GitHub GraphQL API...")
try:
    loader.extract_prs_graphql(limit=500)
except Exception as e:
    print(f"Error extracting PRs: {e}")

print("\n[3/5] Fetching Issues from GitHub GraphQL API...")
try:
    loader.extract_issues_graphql(limit=500)
except Exception as e:
    print(f"Error extracting Issues: {e}")

print("\n[4/5] Exporting raw data cache JSON files...")
try:
    loader.export_data_to_json()
except Exception as e:
    print(f"Error exporting data: {e}")

print("\n[5/5] Chunking text and updating FAISS vector index...")
try:
    embeddings = get_embeddings()
    parser = DocumentParser()
    vectorstore = build_or_update_index(db, parser, embeddings)
    print("Vector store build successfully completed!")
except Exception as e:
    print(f"Error rebuilding vector store: {e}")

print("\n=== COMPLETED SUCCESSFULLY! ===")
