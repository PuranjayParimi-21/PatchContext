import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
from app.database import DatabaseManager
from app.github_loader import GitHubLoader

@pytest.fixture
def mock_loader_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(path)
    yield db, path
    try:
        os.remove(path)
    except OSError:
        pass

@patch('git.Repo')
def test_extract_commits(mock_repo_class, mock_loader_db):
    """Verify GitPython commit parser and issue relationship logic."""
    db, db_path = mock_loader_db
    loader = GitHubLoader(db)
    
    # Mock commit structure
    mock_commit = MagicMock()
    mock_commit.hexsha = "abcdef1234567890"
    mock_commit.author.name = "Git Author"
    mock_commit.committed_datetime.isoformat.return_value = "2026-07-14T21:49:06Z"
    mock_commit.message = "fix: query validation error (#101)"
    mock_commit.parents = []
    
    # Mock git repository iter_commits
    mock_repo = MagicMock()
    mock_repo.iter_commits.return_value = [mock_commit]
    mock_repo_class.clone_from.return_value = mock_repo
    mock_repo_class.return_value = mock_repo
    
    # Execute extraction
    loader.extract_commits(max_count=1)
    
    # Assert commit inserted
    assert db.exists_in_db("commit", "abcdef1234567890") is True
    commit = db.get_item("commit", "abcdef1234567890")
    assert commit["author"] == "Git Author"
    
    # Assert relationship parsed: Commit abcdef1234567890 -> Issue 101
    related = db.get_related_items("commit", "abcdef1234567890")
    assert len(related) == 1
    assert related[0] == ("issue", "101", "references")

@patch('requests.post')
def test_extract_prs_graphql(mock_post, mock_loader_db):
    """Verify GraphQL parser inserts records and merges relations correctly."""
    db, db_path = mock_loader_db
    loader = GitHubLoader(db)
    loader.token = "fake_token"
    
    # Mock GraphQL response payload
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "repository": {
                "pullRequests": {
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": "cursor_end_123"
                    },
                    "nodes": [
                        {
                            "number": 42,
                            "title": "Fix dependency types",
                            "body": "Closes #101",
                            "createdAt": "2026-07-14T20:00:00Z",
                            "state": "MERGED",
                            "merged": True,
                            "mergeCommit": {
                                "oid": "sha_merge_42"
                            },
                            "comments": {
                                "nodes": [{"body": "Nice code!"}]
                            },
                            "reviews": {
                                "nodes": [
                                    {
                                        "comments": {
                                            "nodes": [{"body": "Update this"}]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }
    }
    mock_post.return_value = mock_response
    
    loader.extract_prs_graphql(limit=1)
    
    # Assert PR in database
    assert db.exists_in_db("pr", 42) is True
    pr = db.get_item("pr", 42)
    assert pr["title"] == "Fix dependency types"
    
    # Assert relations: PR merges Commit, PR references Issue
    related = db.get_related_items("pr", 42)
    types = [r[0] for r in related]
    ids = [r[1] for r in related]
    rel_types = [r[2] for r in related]
    
    assert "commit" in types
    assert "sha_merge_42" in ids
    assert "merges" in rel_types
    
    assert "issue" in types
    assert "101" in ids
    assert "references" in rel_types
