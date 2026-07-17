import pytest
from app.verifier import HallucinationGuard
from app.database import DatabaseManager
from langchain_core.documents import Document
import tempfile
import os
from unittest.mock import MagicMock

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

def test_calculate_nli_entailment_mocked():
    """Verify that sentence-level NLI filters citation-carrying sentences based on mocked sequence classification."""
    guard = HallucinationGuard(None)
    
    import torch
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Sentence 1 will be valid (entailment), Sentence 2 will be invalid (contradiction)
    mock_output_valid = MagicMock()
    mock_output_valid.logits = torch.tensor([[0.0, 0.0, 10.0]])  # High index 2 (entailment)
    
    mock_output_invalid = MagicMock()
    mock_output_invalid.logits = torch.tensor([[10.0, 0.0, 0.0]])  # Low index 2
    
    call_count = 0
    def model_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_output_valid
        else:
            return mock_output_invalid
            
    mock_model.side_effect = model_side_effect
    
    guard.tokenizer = mock_tokenizer
    guard.nli_model = mock_model
    guard.device = "cpu"
    
    # We set the NLI threshold explicitly
    from app.config import settings
    settings.nli_entailment_threshold = 0.5
    
    answer = "Sentence one is valid [PR 42]. Sentence two is hallucinated [PR 101]."
    retrieved = [
        Document(page_content="Context showing sentence one is valid.", metadata={"type": "pr", "number": 42, "id": "42"}),
        Document(page_content="Context showing sentence two is hallucinated.", metadata={"type": "pr", "number": 101, "id": "101"})
    ]
    
    filtered_ans, score, _ = guard.calculate_nli_entailment(answer, retrieved)
    
    # Sentence one should remain (and be formatted as markdown link), Sentence two should be removed
    assert "Sentence one is valid" in filtered_ans
    assert "[PR #42](https://github.com/fastapi/fastapi/pull/42)" in filtered_ans or "[PR #42]" in filtered_ans or "PR" in filtered_ans
    assert "Sentence two is hallucinated" not in filtered_ans
    # Softmax on [0, 0, 10] gives ~1.0. Softmax on [10, 0, 0] gives ~0.0. Average is ~0.5
    assert score == pytest.approx(0.5, abs=0.05)

