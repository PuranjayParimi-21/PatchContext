import pytest
from app.verifier import HallucinationGuard
from app.database import DatabaseManager
from langchain_core.documents import Document
import tempfile
import os

@pytest.fixture
def mock_db():
    """Fixture creating database and cleanup."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(path)
    # Seed data
    db.insert_commit("1a2b3c4d5e6f", "Author", "2026-07-14", "feat: commit 1", [], "diff")
    db.insert_pr(42, "PR Title", "PR Body", [], [], "1a2b3c4d5e6f", "2026-07-14")
    db.insert_issue(101, "Issue Title", "Issue Body", [], [], "2026-07-14")
    yield db
    try:
        os.remove(path)
    except OSError:
        pass

def test_parse_citations():
    """Verify that regex matches strict citation syntax tags."""
    guard = HallucinationGuard(None)
    text = "We resolved this in [PR 42] and verified in [Commit 1a2b3c4d5e6f]. Also see [Issue 101]."
    
    parsed = guard.parse_citations(text)
    
    assert "1a2b3c4d5e6f" in parsed["commits"]
    assert "42" in parsed["prs"]
    assert "101" in parsed["issues"]

def test_verify_citations_success(mock_db):
    """Verify that citations existing in both database and retrieved context pass checks."""
    guard = HallucinationGuard(mock_db)
    
    answer = "The issue was discussed in [Issue 101], fixed by [PR 42], and merged in [Commit 1a2b3c4d5e6f]."
    
    # Mock retrieved documents matching the metadata
    retrieved = [
        Document(page_content="PR 42 info", metadata={"type": "pr", "id": "42", "number": 42}),
        Document(page_content="Issue 101 info", metadata={"type": "issue", "id": "101", "number": 101}),
        Document(page_content="Commit 1a2b3c4d5e6f info", metadata={"type": "commit", "id": "1a2b3c4d5e6f", "sha": "1a2b3c4d5e6f"})
    ]
    
    cleaned_ans, verification, _ = guard.verify_citations(answer, retrieved)
    
    # Assert all are verified
    assert verification["prs"]["42"]["verified"] is True
    assert verification["issues"]["101"]["verified"] is True
    assert verification["commits"]["1a2b3c4d5e6f"]["verified"] is True
    
    # The answer should remain unchanged
    assert cleaned_ans == answer

def test_verify_citations_hallucinated(mock_db):
    """Verify that hallucinated citations are flagged and removed from the generated text."""
    guard = HallucinationGuard(mock_db)
    
    # Cited PR 999 and Commit fake999 do not exist in DB or context.
    # Cited Issue 101 exists in DB but is NOT in retrieved context.
    answer = "Fixed in [PR 999] (fake) and [Issue 101] (not in context) and [Commit abc999f12345]."
    
    retrieved = [
        # Only PR 42 in context
        Document(page_content="PR 42 info", metadata={"type": "pr", "id": "42", "number": 42})
    ]
    
    cleaned_ans, verification, _ = guard.verify_citations(answer, retrieved)
    
    # Assert failures
    assert verification["prs"]["999"]["verified"] is False
    assert verification["issues"]["101"]["verified"] is False
    assert verification["commits"]["abc999f12345"]["verified"] is False
    
    # Cleaned answer should have stripped the invalid citation tokens
    assert "[PR 999]" not in cleaned_ans
    assert "[Issue 101]" not in cleaned_ans
    assert "[Commit abc999f12345]" not in cleaned_ans
    assert cleaned_ans == "Fixed in (fake) and (not in context) and."
