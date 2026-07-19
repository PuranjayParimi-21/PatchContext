import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from app.database import DatabaseManager
from app.retriever import HybridRetriever

@pytest.fixture
def mock_db_and_retriever():
    # Setup temporary database
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_path)
    
    # Seed data
    db.insert_commit("2d3e1f0a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e", "Author", "2026-07-14", "feat: commit 1", [], "diff")
    db.insert_pr(145, "PR Title", "PR Body", [], [], "2d3e1f0a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e", "2026-07-14")
    db.insert_issue(458, "Issue Title", "Issue Body", [], [], "2026-07-14")
    db.insert_relationship("pr", "145", "commit", "2d3e1f0a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e", "merges")
    
    # Mark as indexed in SQLite so they are loaded for BM25
    db.mark_as_indexed("commits", "2d3e1f0a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e")
    db.mark_as_indexed("prs", 145)
    db.mark_as_indexed("issues", 458)
    
    # Mock FAISS
    mock_vectorstore = MagicMock()
    
    # Patch CrossEncoder initialization and prediction to avoid model load and download
    with patch("app.retriever.CrossEncoder") as mock_cross_encoder_cls:
        mock_ce = MagicMock()
        mock_ce.predict.side_effect = lambda pairs: [0.9] * len(pairs)
        mock_cross_encoder_cls.return_value = mock_ce
        
        retriever = HybridRetriever(db, mock_vectorstore)
        yield db, retriever, mock_ce
        
    try:
        os.remove(db_path)
    except OSError:
        pass

def test_detect_identifier(mock_db_and_retriever):
    _, retriever, _ = mock_db_and_retriever
    
    # Test PR detection
    assert retriever._detect_identifier("Can you check PR #145?") == ("pr", "145")
    assert retriever._detect_identifier("details of pull request 145") == ("pr", "145")
    
    # Test Issue detection
    assert retriever._detect_identifier("Issue #458 has a bug") == ("issue", "458")
    assert retriever._detect_identifier("what is issue 458") == ("issue", "458")
    
    # Test Commit SHA detection
    assert retriever._detect_identifier("Commit 2d3e1f0a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e details") == ("commit", "2d3e1f0a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e")
    # Prefix check (abbreviated SHA)
    assert retriever._detect_identifier("What was in commit 2d3e1f0?") == ("commit", "2d3e1f0")
    
    # Test non-identifier query
    assert retriever._detect_identifier("How does the app start?") is None

def test_exact_lookup(mock_db_and_retriever):
    db, retriever, _ = mock_db_and_retriever
    
    # Test PR exact lookup
    docs = retriever._exact_lookup("pr", "145")
    assert len(docs) > 0
    assert docs[0].metadata["type"] == "pr"
    assert docs[0].metadata["id"] == "145"
    
    # Test Commit prefix lookup
    docs_commit = retriever._exact_lookup("commit", "2d3e1f0")
    assert len(docs_commit) > 0
    assert docs_commit[0].metadata["type"] == "commit"
    assert docs_commit[0].metadata["id"] == "2d3e1f0a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e"

def test_retrieve_routing_exact(mock_db_and_retriever):
    _, retriever, mock_ce = mock_db_and_retriever
    
    # Mock search functions to check they are NOT called
    retriever._vector_search = MagicMock()
    retriever._bm25_search = MagicMock()
    
    # Run retrieve on exact identifier query
    docs, metrics = retriever.retrieve("Check details for PR #145")
    
    # Verify exact path was taken: vector and bm25 search bypassed
    retriever._vector_search.assert_not_called()
    retriever._bm25_search.assert_not_called()
    
    assert len(docs) > 0
    assert metrics["vector_latency"] == 0.0
    assert metrics["bm25_latency"] == 0.0
    assert "exact_lookup_latency" in metrics

def test_retrieve_routing_hybrid(mock_db_and_retriever):
    _, retriever, mock_ce = mock_db_and_retriever
    
    # Mock search functions
    mock_vector = [MagicMock(page_content="vector doc", metadata={"type": "pr", "id": "10"})]
    mock_bm25 = [MagicMock(page_content="bm25 doc", metadata={"type": "pr", "id": "11"})]
    
    retriever._vector_search = MagicMock(return_value=mock_vector)
    retriever._bm25_search = MagicMock(return_value=mock_bm25)
    
    # Run retrieve on standard query
    docs, metrics = retriever.retrieve("how to initialize the database?")
    
    # Verify hybrid path was taken
    retriever._vector_search.assert_called_once()
    retriever._bm25_search.assert_called_once()
    
    assert len(docs) > 0
    assert metrics["vector_latency"] >= 0.0
    assert metrics["bm25_latency"] >= 0.0
