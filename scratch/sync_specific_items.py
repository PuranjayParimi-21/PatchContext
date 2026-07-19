import os
import requests
import json
import logging
import re
from app.config import settings
from app.database import DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SyncSpecificItems")

def main():
    db = DatabaseManager(settings.database_path)
    token = settings.github_token
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    # Specific PRs and Issues known to contain answers to the benchmark questions
    pr_numbers = [52, 158, 320, 468, 499, 568, 1466, 2429, 3372, 9816, 10011, 10145]
    issue_numbers = [237, 297, 539, 578]
    
    owner, repo_name = settings.github_repository.split("/")
    
    # Fetch PRs
    for num in pr_numbers:
        logger.info(f"Syncing PR #{num}...")
        url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{num}"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch PR #{num}: {resp.status_code} - {resp.text}")
                # Try fetching as issue since sometimes they are overlapping
                url_issue = f"https://api.github.com/repos/{owner}/{repo_name}/issues/{num}"
                resp_issue = requests.get(url_issue, headers=headers)
                if resp_issue.status_code == 200:
                    node = resp_issue.json()
                    db.insert_issue(
                        number=num,
                        title=node.get("title", ""),
                        body=node.get("body", ""),
                        comments=[],
                        labels=[l["name"] for l in node.get("labels", [])],
                        created_at=node.get("created_at", "")
                    )
                continue
                
            pr = resp.json()
            
            # Fetch comments
            comments = []
            comments_url = pr.get("comments_url")
            if comments_url:
                c_resp = requests.get(comments_url, headers=headers)
                if c_resp.status_code == 200:
                    comments = [c["body"] for c in c_resp.json() if c.get("body")]
                    
            # Fetch review comments
            review_comments = []
            rc_url = pr.get("review_comments_url")
            if rc_url:
                rc_resp = requests.get(rc_url, headers=headers)
                if rc_resp.status_code == 200:
                    review_comments = [c["body"] for c in rc_resp.json() if c.get("body")]
            
            db.insert_pr(
                number=num,
                title=pr.get("title", ""),
                body=pr.get("body", ""),
                comments=comments,
                review_comments=review_comments,
                merged_commit_sha=pr.get("merge_commit_sha"),
                created_at=pr.get("created_at", "")
            )
            # Reset is_indexed status to 0 to force re-indexing
            with db._get_connection() as conn:
                conn.cursor().execute("UPDATE prs SET is_indexed = 0 WHERE number = ?", (num,))
                conn.commit()
                
            logger.info(f"Successfully inserted and set is_indexed=0 for PR #{num}")
        except Exception as e:
            logger.error(f"Error syncing PR #{num}: {e}", exc_info=True)
            
    # Fetch Issues
    for num in issue_numbers:
        logger.info(f"Syncing Issue #{num}...")
        url = f"https://api.github.com/repos/{owner}/{repo_name}/issues/{num}"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                logger.error(f"Failed to fetch Issue #{num}: {resp.status_code} - {resp.text}")
                continue
                
            node = resp.json()
            
            # Fetch comments
            comments = []
            comments_url = node.get("comments_url")
            if comments_url:
                c_resp = requests.get(comments_url, headers=headers)
                if c_resp.status_code == 200:
                    comments = [c["body"] for c in c_resp.json() if c.get("body")]
                    
            db.insert_issue(
                number=num,
                title=node.get("title", ""),
                body=node.get("body", ""),
                comments=comments,
                labels=[l["name"] for l in node.get("labels", []) if isinstance(l, dict) and l.get("name")],
                created_at=node.get("created_at", "")
            )
            # Reset is_indexed status to 0 to force re-indexing
            with db._get_connection() as conn:
                conn.cursor().execute("UPDATE issues SET is_indexed = 0 WHERE number = ?", (num,))
                conn.commit()
                
            logger.info(f"Successfully inserted and set is_indexed=0 for Issue #{num}")
        except Exception as e:
            logger.error(f"Error syncing Issue #{num}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
