import os
import re
import json
import logging
import git
import requests
from typing import Dict, List, Any, Optional, Set
from app.config import settings
from app.database import DatabaseManager

logger = logging.getLogger("PatchContext.GithubLoader")

class GitHubLoader:
    """Clones the repository and extracts commits, PRs, and issues using GitPython and GitHub API."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.repo_url = "https://github.com/" + settings.github_repository
        self.token = settings.github_token
        self.owner, self.repo_name = settings.github_repository.split("/")
        
    def clone_or_update_repo(self) -> git.Repo:
        """Clones the FastAPI repo locally or updates it if already cloned."""
        local_path = settings.local_repo_path
        git_dir = os.path.join(local_path, ".git")
        
        if not os.path.exists(local_path) or not os.path.exists(git_dir):
            logger.info(f"Local repository path or .git directory is missing. Cloning {self.repo_url} into {local_path}...")
            if os.path.exists(local_path):
                import shutil
                try:
                    shutil.rmtree(local_path)
                except Exception as e:
                    logger.warning(f"Failed to remove invalid repo path {local_path}: {e}")
            repo = git.Repo.clone_from(self.repo_url, local_path)
            logger.info("Clone completed.")
        else:
            logger.info(f"Opening existing repository at {local_path}...")
            repo = git.Repo(local_path)
            logger.info("Pulling latest updates...")
            try:
                repo.remotes.origin.pull()
                logger.info("Repository updated successfully.")
            except Exception as e:
                logger.warning(f"Failed to pull latest changes (offline or conflict): {e}")
        return repo

    def extract_commits(self, max_count: int = 500) -> None:
        """Extracts commits using GitPython and saves them to the database incrementally."""
        repo = self.clone_or_update_repo()
        logger.info("Extracting commits...")
        
        count = 0
        skipped = 0
        
        # Traverse commits from HEAD
        for commit in repo.iter_commits('HEAD', max_count=max_count):
            sha = commit.hexsha
            
            # Incremental check: stop or skip if we have already indexed this commit
            if self.db.exists_in_db("commit", sha):
                skipped += 1
                if skipped > 10:  # Break if we see many consecutive existing commits
                    logger.info("Encountered existing commits. Stopping commit extraction (incremental update).")
                    break
                continue
                
            skipped = 0
            
            # Find changed files and diffs
            changed_files = []
            diff_text = ""
            try:
                if commit.parents:
                    diffs = commit.parents[0].diff(commit, create_patch=True)
                else:
                    diffs = commit.diff(None, create_patch=True)
                    
                for diff in diffs:
                    a_path = diff.a_path or ""
                    b_path = diff.b_path or ""
                    path = b_path if b_path else a_path
                    if path:
                        changed_files.append(path)
                    
                    if diff.diff:
                        try:
                            patch = diff.diff.decode('utf-8', errors='ignore')
                            if len(diff_text) + len(patch) < 6000:  # Cap diff size per commit to save space
                                diff_text += f"\nFile: {path}\n{patch}"
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Error computing diff for commit {sha[:7]}: {e}")
                
            author = commit.author.name
            date = commit.committed_datetime.isoformat()
            message = commit.message or ""
            
            # Insert commit metadata into SQLite
            self.db.insert_commit(
                sha=sha,
                author=author,
                date=date,
                message=message,
                changed_files=changed_files,
                diff=diff_text
            )
            
            # Parse relationships from commit messages: Commit -> Issue (References)
            issue_refs = re.findall(r'#(\d+)', message)
            for ref in set(issue_refs):
                self.db.insert_relationship("commit", sha, "issue", ref, "references")
                
            count += 1
            if count % 100 == 0:
                logger.info(f"Extracted {count} commits...")
                
        logger.info(f"Commit extraction finished. Added {count} new commits.")

    def _get_graphql_headers(self) -> Dict[str, str]:
        """Prepares headers for GitHub GraphQL/REST requests."""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        else:
            logger.warning("No GITHUB_TOKEN set. GitHub API calls will be heavily rate-limited.")
        return headers

    def extract_prs_graphql(self, limit: int = 100) -> None:
        """Extracts PRs using GitHub GraphQL API, falling back to REST API if token is invalid/empty."""
        if not self.token:
            logger.warning("No GITHUB_TOKEN configured. Falling back to REST API for PR extraction.")
            self.extract_prs_rest(limit=limit)
            return
            
        logger.info("Extracting Pull Requests using GraphQL API...")
        url = "https://api.github.com/graphql"
        headers = self._get_graphql_headers()
        
        # Load pagination cursor from state
        cursor = self.db.get_extraction_state("graphql_prs_cursor", None)
        has_next = True
        total_fetched = 0
        
        query = """
        query($owner: String!, $name: String!, $cursor: String) {
          repository(owner: $owner, name: $name) {
            pullRequests(first: 30, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                number
                title
                body
                createdAt
                state
                merged
                mergeCommit {
                  oid
                }
                comments(first: 10) {
                  nodes {
                    body
                  }
                }
                reviews(first: 10) {
                  nodes {
                    comments(first: 5) {
                      nodes {
                        body
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        while has_next and total_fetched < limit:
            variables = {"owner": self.owner, "name": self.repo_name, "cursor": cursor}
            try:
                response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
                if response.status_code == 403 or response.status_code == 401:
                    logger.error(f"GraphQL authentication or rate limit error ({response.status_code}). Falling back to REST.")
                    self.extract_prs_rest(limit - total_fetched)
                    return
                elif response.status_code != 200:
                    logger.error(f"GraphQL request failed with code {response.status_code}: {response.text}")
                    break
                    
                res_data = response.json()
                if "errors" in res_data:
                    logger.error(f"GraphQL returned errors: {res_data['errors']}")
                    break
                    
                repo_data = res_data.get("data", {}).get("repository", {})
                if not repo_data:
                    logger.warning("No repository data returned. Token might lack permissions.")
                    break
                    
                pr_connection = repo_data.get("pullRequests", {})
                nodes = pr_connection.get("nodes", [])
                
                for node in nodes:
                    number = node["number"]
                    title = node["title"] or ""
                    body = node["body"] or ""
                    created_at = node["createdAt"] or ""
                    
                    merged_sha = None
                    if node.get("merged") and node.get("mergeCommit"):
                        merged_sha = node["mergeCommit"]["oid"]
                        
                    # Extract comments
                    comments = [c["body"] for c in node.get("comments", {}).get("nodes", []) if c.get("body")]
                    
                    # Extract review comments
                    review_comments = []
                    for review in node.get("reviews", {}).get("nodes", []):
                        for rc in review.get("comments", {}).get("nodes", []):
                            if rc.get("body"):
                                review_comments.append(rc["body"])
                                
                    # Save PR
                    self.db.insert_pr(
                        number=number,
                        title=title,
                        body=body,
                        comments=comments,
                        review_comments=review_comments,
                        merged_commit_sha=merged_sha,
                        created_at=created_at
                    )
                    
                    # Store relationship: PR -> Merge Commit (if exists)
                    if merged_sha:
                        self.db.insert_relationship("pr", str(number), "commit", merged_sha, "merges")
                        
                    # Parse relationships from PR body and comments: PR -> Issue (references/closes)
                    pr_text = f"{title}\n{body}\n" + "\n".join(comments)
                    issue_refs = re.findall(r'#(\d+)', pr_text)
                    for ref in set(issue_refs):
                        self.db.insert_relationship("pr", str(number), "issue", ref, "references")
                        
                    total_fetched += 1
                    
                page_info = pr_connection.get("pageInfo", {})
                has_next = page_info.get("hasNextPage", False)
                cursor = page_info.get("endCursor", None)
                
                # Checkpoint pagination
                if cursor:
                    self.db.set_extraction_state("graphql_prs_cursor", cursor)
                    
                logger.info(f"GraphQL: Fetched {total_fetched} PRs so far. Cursor: {cursor}")
                
            except Exception as e:
                logger.error(f"Error fetching PRs via GraphQL: {e}", exc_info=True)
                break
                
        logger.info(f"GraphQL PR extraction completed. Fetched {total_fetched} items.")

    def extract_issues_graphql(self, limit: int = 100) -> None:
        """Extracts Issues using GitHub GraphQL API, falling back to REST if needed."""
        if not self.token:
            logger.warning("No GITHUB_TOKEN configured. Falling back to REST API for Issues extraction.")
            self.extract_issues_rest(limit=limit)
            return
            
        logger.info("Extracting Issues using GraphQL API...")
        url = "https://api.github.com/graphql"
        headers = self._get_graphql_headers()
        
        cursor = self.db.get_extraction_state("graphql_issues_cursor", None)
        has_next = True
        total_fetched = 0
        
        query = """
        query($owner: String!, $name: String!, $cursor: String) {
          repository(owner: $owner, name: $name) {
            issues(first: 30, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                number
                title
                body
                createdAt
                labels(first: 10) {
                  nodes {
                    name
                  }
                }
                comments(first: 15) {
                  nodes {
                    body
                  }
                }
              }
            }
          }
        }
        """
        
        while has_next and total_fetched < limit:
            variables = {"owner": self.owner, "name": self.repo_name, "cursor": cursor}
            try:
                response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
                if response.status_code == 403 or response.status_code == 401:
                    logger.error(f"GraphQL authentication or rate limit error ({response.status_code}). Falling back to REST.")
                    self.extract_issues_rest(limit - total_fetched)
                    return
                elif response.status_code != 200:
                    logger.error(f"GraphQL request failed with code {response.status_code}: {response.text}")
                    break
                    
                res_data = response.json()
                if "errors" in res_data:
                    logger.error(f"GraphQL returned errors: {res_data['errors']}")
                    break
                    
                repo_data = res_data.get("data", {}).get("repository", {})
                issue_connection = repo_data.get("issues", {})
                nodes = issue_connection.get("nodes", [])
                
                for node in nodes:
                    number = node["number"]
                    title = node["title"] or ""
                    body = node["body"] or ""
                    created_at = node["createdAt"] or ""
                    
                    labels = [l["name"] for l in node.get("labels", {}).get("nodes", []) if l.get("name")]
                    comments = [c["body"] for c in node.get("comments", {}).get("nodes", []) if c.get("body")]
                    
                    # Save Issue
                    self.db.insert_issue(
                        number=number,
                        title=title,
                        body=body,
                        comments=comments,
                        labels=labels,
                        created_at=created_at
                    )
                    
                    # Parse relationships from Issue body: Issue -> other Issue references
                    issue_text = f"{title}\n{body}\n" + "\n".join(comments)
                    other_refs = re.findall(r'#(\d+)', issue_text)
                    for ref in set(other_refs):
                        if int(ref) != number:
                            self.db.insert_relationship("issue", str(number), "issue", ref, "references")
                            
                    total_fetched += 1
                    
                page_info = issue_connection.get("pageInfo", {})
                has_next = page_info.get("hasNextPage", False)
                cursor = page_info.get("endCursor", None)
                
                if cursor:
                    self.db.set_extraction_state("graphql_issues_cursor", cursor)
                    
                logger.info(f"GraphQL: Fetched {total_fetched} Issues so far. Cursor: {cursor}")
                
            except Exception as e:
                logger.error(f"Error fetching Issues via GraphQL: {e}", exc_info=True)
                break
                
        logger.info(f"GraphQL Issue extraction completed. Fetched {total_fetched} items.")

    def extract_prs_rest(self, limit: int = 100) -> None:
        """REST API fallback to extract Pull Requests with pagination."""
        logger.info("Extracting PRs via GitHub REST API...")
        headers = self._get_graphql_headers()
        url = f"https://api.github.com/repos/{settings.github_repository}/pulls"
        
        # Load checkpoint: highest number we've inserted, or start page
        start_page = int(self.db.get_extraction_state("rest_prs_page", "1"))
        page = start_page
        total_fetched = 0
        
        while total_fetched < limit:
            params = {"state": "all", "sort": "created", "direction": "asc", "per_page": 50, "page": page}
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code != 200:
                    logger.error(f"REST API Pulls failed: {response.status_code} - {response.text}")
                    break
                    
                prs_data = response.json()
                if not prs_data:
                    logger.info("No more PRs found on REST page.")
                    break
                    
                for pr in prs_data:
                    if total_fetched >= limit:
                        break
                        
                    number = pr["number"]
                    title = pr["title"] or ""
                    body = pr["body"] or ""
                    created_at = pr["created_at"] or ""
                    merged_sha = pr.get("merge_commit_sha")
                    
                    # Fetch comments using comments url
                    comments = []
                    comments_url = pr.get("comments_url")
                    if comments_url:
                        comments_resp = requests.get(comments_url, headers=headers)
                        if comments_resp.status_code == 200:
                            comments = [c["body"] for c in comments_resp.json() if c.get("body")]
                            
                    # For REST we can fetch review comments as well (optional fallback)
                    review_comments = []
                    review_comments_url = pr.get("review_comments_url")
                    if review_comments_url:
                        rc_resp = requests.get(review_comments_url, headers=headers)
                        if rc_resp.status_code == 200:
                            review_comments = [c["body"] for c in rc_resp.json() if c.get("body")]
                            
                    # Save PR
                    self.db.insert_pr(
                        number=number,
                        title=title,
                        body=body,
                        comments=comments,
                        review_comments=review_comments,
                        merged_commit_sha=merged_sha,
                        created_at=created_at
                    )
                    
                    # Insert merges relation
                    if merged_sha:
                        self.db.insert_relationship("pr", str(number), "commit", merged_sha, "merges")
                        
                    # Relationships
                    pr_text = f"{title}\n{body}\n" + "\n".join(comments)
                    issue_refs = re.findall(r'#(\d+)', pr_text)
                    for ref in set(issue_refs):
                        self.db.insert_relationship("pr", str(number), "issue", ref, "references")
                        
                    total_fetched += 1
                    
                page += 1
                self.db.set_extraction_state("rest_prs_page", str(page))
                logger.info(f"REST: Fetched {total_fetched} PRs. Next page: {page}")
                
            except Exception as e:
                logger.error(f"Error fetching PRs via REST API: {e}", exc_info=True)
                break

    def extract_issues_rest(self, limit: int = 100) -> None:
        """REST API fallback to extract Issues (filtering out PRs) with pagination."""
        logger.info("Extracting Issues via GitHub REST API...")
        headers = self._get_graphql_headers()
        url = f"https://api.github.com/repos/{settings.github_repository}/issues"
        
        start_page = int(self.db.get_extraction_state("rest_issues_page", "1"))
        page = start_page
        total_fetched = 0
        
        while total_fetched < limit:
            params = {"state": "all", "sort": "created", "direction": "asc", "per_page": 50, "page": page}
            try:
                response = requests.get(url, params=params, headers=headers)
                if response.status_code != 200:
                    logger.error(f"REST API Issues failed: {response.status_code} - {response.text}")
                    break
                    
                issues_data = response.json()
                if not issues_data:
                    logger.info("No more issues found on REST page.")
                    break
                    
                for issue in issues_data:
                    # Skip pull requests (GitHub REST API returns both issues and PRs on this endpoint)
                    if "pull_request" in issue:
                        continue
                        
                    if total_fetched >= limit:
                        break
                        
                    number = issue["number"]
                    title = issue["title"] or ""
                    body = issue["body"] or ""
                    created_at = issue["created_at"] or ""
                    
                    labels = [l["name"] for l in issue.get("labels", []) if l.get("name")]
                    
                    comments = []
                    comments_url = issue.get("comments_url")
                    if comments_url:
                        comments_resp = requests.get(comments_url, headers=headers)
                        if comments_resp.status_code == 200:
                            comments = [c["body"] for c in comments_resp.json() if c.get("body")]
                            
                    # Save Issue
                    self.db.insert_issue(
                        number=number,
                        title=title,
                        body=body,
                        comments=comments,
                        labels=labels,
                        created_at=created_at
                    )
                    
                    # Relationships
                    issue_text = f"{title}\n{body}\n" + "\n".join(comments)
                    other_refs = re.findall(r'#(\d+)', issue_text)
                    for ref in set(other_refs):
                        if int(ref) != number:
                            self.db.insert_relationship("issue", str(number), "issue", ref, "references")
                            
                    total_fetched += 1
                    
                page += 1
                self.db.set_extraction_state("rest_issues_page", str(page))
                logger.info(f"REST: Fetched {total_fetched} Issues. Next page: {page}")
                
            except Exception as e:
                logger.error(f"Error fetching Issues via REST API: {e}", exc_info=True)
                break
                
    def export_data_to_json(self) -> None:
        """
        Exports all extracted SQLite rows into data/commits.json, data/prs.json, data/issues.json
        to comply with Module 1 JSON storage requirement.
        """
        os.makedirs("data", exist_ok=True)
        
        # Helper to export table
        def export_table(table_name: str, file_path: str):
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = [dict(row) for row in cursor.fetchall()]
                
                # De-serialize JSON fields
                for r in rows:
                    for key in ("changed_files", "comments", "review_comments", "labels"):
                        if key in r and r[key]:
                            try:
                                r[key] = json.loads(r[key])
                            except Exception:
                                pass
                                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2, ensure_ascii=False)
                logger.info(f"Exported {len(rows)} records to {file_path}")

        export_table("commits", "data/commits.json")
        export_table("prs", "data/prs.json")
        export_table("issues", "data/issues.json")
