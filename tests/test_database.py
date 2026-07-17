import os
import pytest
import tempfile
from app.database import DatabaseManager

@pytest.fixture
def temp_db():
    """Fixture to create a temporary database file and clean it up afterward."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(path)
    yield db
    # Cleanup database files
    try:
        os.remove(path)
    except OSError:
        pass
    
    # Clean WAL files if any
    for ext in ("-wal", "-shm"):
        if os.path.exists(path + ext):
            try:
                os.remove(path + ext)
            except OSError:
                pass

def test_database_init(temp_db):
    """Verify that stats show 0 counts on fresh database initialization."""
    stats = temp_db.get_stats()
    assert stats["commits"] == 0
    assert stats["prs"] == 0
    assert stats["issues"] == 0
    assert stats["relationships"] == 0

def test_commit_operations(temp_db):
    """Verify insert, get, search, and indexing operations on commits."""
    sha = "abc123def456"
    temp_db.insert_commit(
        sha=sha,
        author="Test Author",
        date="2026-07-14T21:49:06Z",
        message="test: add sqlite database tests",
        changed_files=["app/database.py", "tests/test_database.py"],
        diff="--- a/app/database.py\n+++ b/app/database.py\n..."
    )
    
    # Check if exists
    assert temp_db.exists_in_db("commit", sha) is True
    assert temp_db.exists_in_db("commit", "nonexistent") is False
    
    # Fetch commit
    commit = temp_db.get_item("commit", sha)
    assert commit is not None
    assert commit["author"] == "Test Author"
    assert commit["is_indexed"] == 0
    
    # Check unindexed
    unindexed = temp_db.get_unindexed_items("commits")
    assert len(unindexed) == 1
    assert unindexed[0]["sha"] == sha
    
    # Mark indexed
    temp_db.mark_as_indexed("commits", sha)
    unindexed = temp_db.get_unindexed_items("commits")
    assert len(unindexed) == 0
    
    # Check updated status
    commit = temp_db.get_item("commit", sha)
    assert commit["is_indexed"] == 1

def test_pr_and_issue_operations(temp_db):
    """Verify insert and fetch operations for pull requests and issues."""
    # Test PR
    pr_num = 42
    temp_db.insert_pr(
        number=pr_num,
        title="Fix database WAL mode bug",
        body="This fixes a connection error in sqlite WAL mode.",
        comments=["Nice fix!", "LGTM"],
        review_comments=["Looks good"],
        merged_commit_sha="abc123def456",
        created_at="2026-07-14T20:00:00Z"
    )
    assert temp_db.exists_in_db("pr", pr_num) is True
    
    pr = temp_db.get_item("pr", pr_num)
    assert pr is not None
    assert pr["title"] == "Fix database WAL mode bug"
    assert pr["merged_commit_sha"] == "abc123def456"
    
    # Test Issue
    issue_num = 101
    temp_db.insert_issue(
        number=issue_num,
        title="WAL mode connection error",
        body="Database fails under heavy concurrent writes.",
        comments=["Confirming this issue.", "Working on a PR."],
        labels=["bug", "database"],
        created_at="2026-07-14T19:00:00Z"
    )
    assert temp_db.exists_in_db("issue", issue_num) is True
    
    issue = temp_db.get_item("issue", issue_num)
    assert issue is not None
    assert issue["title"] == "WAL mode connection error"

def test_relationships(temp_db):
    """Verify inserting and retrieving relations (graph links)."""
    # Create test relationships
    # PR 42 merges commit sha 'abc'
    temp_db.insert_relationship("pr", "42", "commit", "abc123def456", "merges")
    # PR 42 closes issue 101
    temp_db.insert_relationship("pr", "42", "issue", "101", "closes")
    
    # Retrieve related for PR 42
    related = temp_db.get_related_items("pr", "42")
    assert len(related) == 2
    # Check formats
    types = [r[0] for r in related]
    ids = [r[1] for r in related]
    rel_types = [r[2] for r in related]
    
    assert "commit" in types
    assert "issue" in types
    assert "abc123def456" in ids
    assert "101" in ids
    assert "merges" in rel_types
    assert "closes" in rel_types
    
    # Check inverse relationships
    related_to_issue = temp_db.get_related_items("issue", "101")
    assert len(related_to_issue) == 1
    assert related_to_issue[0] == ("pr", "42", "inverse_closes")

def test_extraction_state(temp_db):
    """Verify save and resume capabilities using state mapping."""
    temp_db.set_extraction_state("last_pr_cursor", "cursor_xyz_123")
    val = temp_db.get_extraction_state("last_pr_cursor")
    assert val == "cursor_xyz_123"
    
    # Default fallback
    assert temp_db.get_extraction_state("non_existent_key", "default_val") == "default_val"
