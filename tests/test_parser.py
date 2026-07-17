import json
import pytest
from app.parser import DocumentParser

def test_parse_commit():
    """Verify that commit rows are parsed with proper page content format and metadata keys."""
    parser = DocumentParser()
    row = {
        "sha": "1a2b3c4d5e6f7g8h9i0j",
        "author": "Commit Author",
        "date": "2026-07-14T21:49:06Z",
        "message": "feat: initialize repository parser\n\nMore descriptions here.",
        "changed_files": json.dumps(["app/parser.py", "README.md"]),
        "diff": "+++ b/app/parser.py\n+class DocumentParser:"
    }
    
    doc = parser.parse_commit(row)
    
    assert doc.metadata["type"] == "commit"
    assert doc.metadata["sha"] == "1a2b3c4d5e6f7g8h9i0j"
    assert doc.metadata["id"] == "1a2b3c4d5e6f7g8h9i0j"
    assert doc.metadata["author"] == "Commit Author"
    assert doc.metadata["title"] == "feat: initialize repository parser"
    
    assert "Type: Commit" in doc.page_content
    assert "SHA: 1a2b3c4d5e6f7g8h9i0j" in doc.page_content
    assert "app/parser.py, README.md" in doc.page_content
    assert "class DocumentParser" in doc.page_content

def test_parse_pr():
    """Verify that PR rows are parsed with comments, reviews, merge commits, and metadata."""
    parser = DocumentParser()
    row = {
        "number": 100,
        "title": "Fix typing in dependencies",
        "body": "This PR corrects type annotations in fastapi dependencies.",
        "comments": json.dumps(["Nice addition", "Thanks!"]),
        "review_comments": json.dumps(["Fix this line"]),
        "merged_commit_sha": "merge_sha_123",
        "created_at": "2026-07-14T12:00:00Z"
    }
    
    doc = parser.parse_pr(row)
    
    assert doc.metadata["type"] == "pr"
    assert doc.metadata["number"] == 100
    assert doc.metadata["id"] == "100"
    assert doc.metadata["title"] == "Fix typing in dependencies"
    assert doc.metadata["merged_commit_sha"] == "merge_sha_123"
    
    assert "Type: Pull Request" in doc.page_content
    assert "PR Number: #100" in doc.page_content
    assert "- Comment: Nice addition" in doc.page_content
    assert "- Review: Fix this line" in doc.page_content
    assert "Merged Commit SHA: merge_sha_123" in doc.page_content

def test_parse_issue():
    """Verify that issue rows are parsed with comments, labels, and metadata."""
    parser = DocumentParser()
    row = {
        "number": 50,
        "title": "Bug in query parameters validation",
        "body": "Query parameters are not validated if declared as list.",
        "comments": json.dumps(["Confirming this", "Can we write a test?"]),
        "labels": json.dumps(["bug", "validation"]),
        "created_at": "2026-07-14T10:00:00Z"
    }
    
    doc = parser.parse_issue(row)
    
    assert doc.metadata["type"] == "issue"
    assert doc.metadata["number"] == 50
    assert doc.metadata["id"] == "50"
    assert doc.metadata["labels"] == "bug, validation"
    
    assert "Type: Issue" in doc.page_content
    assert "Issue Number: #50" in doc.page_content
    assert "Labels: bug, validation" in doc.page_content
    assert "- Comment: Confirming this" in doc.page_content
