import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.database import DatabaseManager

logger = logging.getLogger("PatchContext.Parser")

class DocumentParser:
    """Converts SQLite database rows into LangChain Document objects and chunks them."""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def parse_commit(self, row: Dict[str, Any]) -> Document:
        """Converts a database commit row to a LangChain Document."""
        sha = row["sha"]
        author = row["author"] or "Unknown"
        date = row["date"] or "Unknown"
        message = row["message"] or ""
        
        # De-serialize changed files if string
        files = row["changed_files"]
        if isinstance(files, str):
            try:
                files = json.loads(files)
            except Exception:
                files = []
        files_str = ", ".join(files) if files else "None"
        
        diff = row["diff"] or "No diff available."
        
        # Structure the content clearly for embedding and LLM comprehension
        content = (
            f"Type: Commit\n"
            f"SHA: {sha}\n"
            f"Author: {author}\n"
            f"Date: {date}\n"
            f"Message: {message}\n"
            f"Changed Files: {files_str}\n"
            f"Diff:\n{diff}"
        )
        
        metadata = {
            "type": "commit",
            "id": sha,
            "sha": sha,
            "author": author,
            "date": date,
            "title": message.split("\n")[0]
        }
        return Document(page_content=content, metadata=metadata)

    def parse_pr(self, row: Dict[str, Any]) -> Document:
        """Converts a database pull request row to a LangChain Document."""
        number = row["number"]
        title = row["title"] or ""
        body = row["body"] or ""
        created_at = row["created_at"] or "Unknown"
        merged_sha = row["merged_commit_sha"] or "Not Merged"
        
        # De-serialize comments
        comments = row["comments"]
        if isinstance(comments, str):
            try:
                comments = json.loads(comments)
            except Exception:
                comments = []
        comments_str = "\n".join([f"- Comment: {c}" for c in comments]) if comments else "None"
        
        # De-serialize review comments
        review_comments = row["review_comments"]
        if isinstance(review_comments, str):
            try:
                review_comments = json.loads(review_comments)
            except Exception:
                review_comments = []
        rev_comments_str = "\n".join([f"- Review: {r}" for r in review_comments]) if review_comments else "None"
        
        content = (
            f"Type: Pull Request\n"
            f"PR Number: #{number}\n"
            f"Title: {title}\n"
            f"Description: {body}\n"
            f"Merged Commit SHA: {merged_sha}\n"
            f"Created At: {created_at}\n"
            f"Discussion Comments:\n{comments_str}\n"
            f"Code Review Comments:\n{rev_comments_str}"
        )
        
        metadata = {
            "type": "pr",
            "id": str(number),
            "number": number,
            "title": title,
            "merged_commit_sha": merged_sha,
            "date": created_at
        }
        return Document(page_content=content, metadata=metadata)

    def parse_issue(self, row: Dict[str, Any]) -> Document:
        """Converts a database issue row to a LangChain Document."""
        number = row["number"]
        title = row["title"] or ""
        body = row["body"] or ""
        created_at = row["created_at"] or "Unknown"
        
        # Labels
        labels = row["labels"]
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except Exception:
                labels = []
        labels_str = ", ".join(labels) if labels else "None"
        
        # Comments
        comments = row["comments"]
        if isinstance(comments, str):
            try:
                comments = json.loads(comments)
            except Exception:
                comments = []
        comments_str = "\n".join([f"- Comment: {c}" for c in comments]) if comments else "None"
        
        content = (
            f"Type: Issue\n"
            f"Issue Number: #{number}\n"
            f"Title: {title}\n"
            f"Body: {body}\n"
            f"Labels: {labels_str}\n"
            f"Created At: {created_at}\n"
            f"Discussion Comments:\n{comments_str}"
        )
        
        metadata = {
            "type": "issue",
            "id": str(number),
            "number": number,
            "title": title,
            "labels": labels_str,
            "date": created_at
        }
        return Document(page_content=content, metadata=metadata)

    def process_unindexed_items(self, db: DatabaseManager) -> List[Document]:
        """Queries the DB for all unindexed items, parses them to Documents, and splits them into chunks."""
        raw_documents = []
        
        # Commits
        unindexed_commits = db.get_unindexed_items("commits")
        logger.info(f"Parsing {len(unindexed_commits)} unindexed commits.")
        for row in unindexed_commits:
            doc = self.parse_commit(row)
            raw_documents.append(doc)
            
        # Pull Requests
        unindexed_prs = db.get_unindexed_items("prs")
        logger.info(f"Parsing {len(unindexed_prs)} unindexed PRs.")
        for row in unindexed_prs:
            doc = self.parse_pr(row)
            raw_documents.append(doc)
            
        # Issues
        unindexed_issues = db.get_unindexed_items("issues")
        logger.info(f"Parsing {len(unindexed_issues)} unindexed issues.")
        for row in unindexed_issues:
            doc = self.parse_issue(row)
            raw_documents.append(doc)
            
        if not raw_documents:
            logger.info("No new documents to process.")
            return []
            
        # Chunk the raw documents
        chunks = self.splitter.split_documents(raw_documents)
        logger.info(f"Split {len(raw_documents)} source documents into {len(chunks)} text chunks.")
        
        return chunks
