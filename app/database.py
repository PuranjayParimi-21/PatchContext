import sqlite3
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("PatchContext.Database")

class DatabaseManager:
    """Manages the SQLite database for commits, PRs, issues, and relationships."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a robust SQLite connection with timeout to prevent DatabaseError on Streamlit Cloud."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,           # Wait up to 30s for locks to clear before raising an error
            check_same_thread=False  # Allow cross-thread use inside cache_resource
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        logger.info(f"Initializing database at {self.db_path}")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create commits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS commits (
                    sha TEXT PRIMARY KEY,
                    author TEXT,
                    date TEXT,
                    message TEXT,
                    changed_files TEXT,
                    diff TEXT,
                    is_indexed INTEGER DEFAULT 0
                )
            """)
            
            # Create Pull Requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prs (
                    number INTEGER PRIMARY KEY,
                    title TEXT,
                    body TEXT,
                    comments TEXT,
                    review_comments TEXT,
                    merged_commit_sha TEXT,
                    created_at TEXT,
                    is_indexed INTEGER DEFAULT 0
                )
            """)
            
            # Create Issues table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS issues (
                    number INTEGER PRIMARY KEY,
                    title TEXT,
                    body TEXT,
                    comments TEXT,
                    labels TEXT,
                    created_at TEXT,
                    is_indexed INTEGER DEFAULT 0
                )
            """)
            
            # Create Relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT,
                    source_id TEXT,
                    target_type TEXT,
                    target_id TEXT,
                    rel_type TEXT,
                    UNIQUE(source_type, source_id, target_type, target_id, rel_type)
                )
            """)
            
            # Create Extraction State table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extraction_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Optimize DB performance
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            
            conn.commit()
        logger.info("Database tables initialized successfully.")

    def insert_commit(self, sha: str, author: str, date: str, message: str, 
                      changed_files: List[str], diff: str) -> None:
        """Inserts or updates a commit record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO commits (sha, author, date, message, changed_files, diff)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sha) DO UPDATE SET
                        author=excluded.author,
                        date=excluded.date,
                        message=excluded.message,
                        changed_files=excluded.changed_files,
                        diff=excluded.diff
                """, (sha, author, date, message, json.dumps(changed_files), diff))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting commit {sha[:7]}: {e}", exc_info=True)
            raise

    def insert_pr(self, number: int, title: str, body: str, comments: List[str], 
                  review_comments: List[str], merged_commit_sha: Optional[str], 
                  created_at: str) -> None:
        """Inserts or updates a pull request record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO prs (number, title, body, comments, review_comments, merged_commit_sha, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(number) DO UPDATE SET
                        title=excluded.title,
                        body=excluded.body,
                        comments=excluded.comments,
                        review_comments=excluded.review_comments,
                        merged_commit_sha=excluded.merged_commit_sha,
                        created_at=excluded.created_at
                """, (number, title, body, json.dumps(comments), json.dumps(review_comments), 
                      merged_commit_sha, created_at))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting PR #{number}: {e}", exc_info=True)
            raise

    def insert_issue(self, number: int, title: str, body: str, comments: List[str], 
                     labels: List[str], created_at: str) -> None:
        """Inserts or updates an issue record."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO issues (number, title, body, comments, labels, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(number) DO UPDATE SET
                        title=excluded.title,
                        body=excluded.body,
                        comments=excluded.comments,
                        labels=excluded.labels,
                        created_at=excluded.created_at
                """, (number, title, body, json.dumps(comments), json.dumps(labels), created_at))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting Issue #{number}: {e}", exc_info=True)
            raise

    def insert_relationship(self, source_type: str, source_id: str, 
                            target_type: str, target_id: str, rel_type: str) -> None:
        """Inserts a directed relationship between two entities."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO relationships (source_type, source_id, target_type, target_id, rel_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (source_type, str(source_id), target_type, str(target_id), rel_type))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting relationship {source_type}:{source_id} -> {target_type}:{target_id}: {e}")

    def get_unindexed_items(self, item_type: str) -> List[Dict[str, Any]]:
        """Retrieves items of type 'commits', 'prs', or 'issues' that are not yet indexed."""
        if item_type not in ("commits", "prs", "issues"):
            raise ValueError(f"Invalid item type: {item_type}")
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {item_type} WHERE is_indexed = 0")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def mark_as_indexed(self, item_type: str, item_id: Any) -> None:
        """Marks a commit (sha) or PR/issue (number) as indexed."""
        table = item_type
        if table not in ("commits", "prs", "issues"):
            raise ValueError(f"Invalid item type: {item_type}")
            
        id_col = "sha" if table == "commits" else "number"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE {table} SET is_indexed = 1 WHERE {id_col} = ?", (str(item_id),))
            conn.commit()

    def exists_in_db(self, item_type: str, item_id: Any) -> bool:
        """Checks if a commit, PR, or issue exists in the database. Returns False safely on any error."""
        table = item_type
        if table == "commit":
            table = "commits"
        elif table == "pr":
            table = "prs"
        elif table == "issue":
            table = "issues"
            
        if table not in ("commits", "prs", "issues"):
            return False
            
        id_col = "sha" if table == "commits" else "number"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT 1 FROM {table} WHERE {id_col} = ? LIMIT 1", (str(item_id),))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking existence of {item_type} '{item_id}' in database: {e}")
            return False

    def get_item(self, item_type: str, item_id: Any) -> Optional[Dict[str, Any]]:
        """Retrieves a single commit, PR, or issue record. Returns None safely on any error."""
        table = item_type
        if table == "commit":
            table = "commits"
        elif table == "pr":
            table = "prs"
        elif table == "issue":
            table = "issues"
            
        if table not in ("commits", "prs", "issues"):
            return None
            
        id_col = "sha" if table == "commits" else "number"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table} WHERE {id_col} = ?", (str(item_id),))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error retrieving {item_type} '{item_id}' from database: {e}")
            return None

    def get_related_items(self, item_type: str, item_id: Any) -> List[Tuple[str, str, str]]:
        """
        Retrieves all items related to the given entity.
        Returns a list of tuples: (related_type, related_id, relationship_type)
        """
        item_id_str = str(item_id)
        related = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Relationships where current item is source
            cursor.execute("""
                SELECT target_type, target_id, rel_type FROM relationships 
                WHERE source_type = ? AND source_id = ?
            """, (item_type, item_id_str))
            for row in cursor.fetchall():
                related.append((row['target_type'], row['target_id'], row['rel_type']))
                
            # Relationships where current item is target
            cursor.execute("""
                SELECT source_type, source_id, rel_type FROM relationships 
                WHERE target_type = ? AND target_id = ?
            """, (item_type, item_id_str))
            for row in cursor.fetchall():
                related.append((row['source_type'], row['source_id'], f"inverse_{row['rel_type']}"))
                
        return related

    def set_extraction_state(self, key: str, value: str) -> None:
        """Saves a string configuration state for resumable tasks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO extraction_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, value))
            conn.commit()

    def get_extraction_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves a saved extraction state value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM extraction_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row['value'] if row else default

    def get_stats(self) -> Dict[str, int]:
        """Returns counts of commits, PRs, issues, and relationships."""
        stats = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for table in ("commits", "prs", "issues", "relationships"):
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
        return stats
