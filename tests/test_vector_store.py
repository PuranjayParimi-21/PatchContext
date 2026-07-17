import os
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock
from langchain_core.embeddings import Embeddings
from app.database import DatabaseManager
from app.parser import DocumentParser
from app.vector_store import load_vector_store, build_or_update_index
from app.config import settings

class MockEmbeddings(Embeddings):
    """Mock embeddings helper to test vector stores offline without calling OpenAI."""
    def embed_documents(self, texts):
        # text-embedding-3-small returns 1536-dimensional vectors
        return [[0.1] * 1536 for _ in texts]
        
    def embed_query(self, text):
        return [0.1] * 1536

@pytest.fixture
def test_env():
    """Fixture to create temp database and temp vectorstore directories."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    vs_dir = tempfile.mkdtemp()
    
    # Store settings backups
    orig_db = settings.database_path
    orig_vs = settings.vectorstore_path
    
    # Set to temp paths
    settings.database_path = db_path
    settings.vectorstore_path = vs_dir
    
    db = DatabaseManager(db_path)
    # Seed data
    db.insert_commit("sha1", "Author", "2026-07-14", "feat: commit 1", [], "diff")
    db.insert_pr(42, "PR Title", "PR Body", [], [], "sha1", "2026-07-14")
    
    yield db, vs_dir
    
    # Restore settings
    settings.database_path = orig_db
    settings.vectorstore_path = orig_vs
    
    # Cleanups
    try:
        os.remove(db_path)
        shutil.rmtree(vs_dir)
    except OSError:
        pass

def test_load_vector_store_missing(test_env):
    """Verify loading a nonexistent store returns None."""
    _, vs_dir = test_env
    embeddings = MockEmbeddings()
    # Path is empty, so should return None
    assert load_vector_store(embeddings) is None

def test_build_or_update_index(test_env):
    """Verify that build_or_update_index chunks documents, adds to FAISS, saves, and updates DB status."""
    db, vs_dir = test_env
    embeddings = MockEmbeddings()
    parser = DocumentParser()
    
    # Run indexing
    vectorstore = build_or_update_index(db, parser, embeddings)
    
    assert vectorstore is not None
    # Index files should have been created
    assert os.path.exists(os.path.join(vs_dir, "index.faiss"))
    assert os.path.exists(os.path.join(vs_dir, "index.pkl"))
    
    # Source rows in SQLite should be marked indexed
    commit = db.get_item("commit", "sha1")
    pr = db.get_item("pr", 42)
    assert commit["is_indexed"] == 1
    assert pr["is_indexed"] == 1
    
    # A subsequent call should find no unindexed items and return same store
    vectorstore_second = build_or_update_index(db, parser, embeddings)
    assert vectorstore_second is not None
