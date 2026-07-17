from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from app.config import settings
from app.database import DatabaseManager
from app.github_loader import GitHubLoader
from app.parser import DocumentParser
from app.embeddings import get_embeddings
from app.vector_store import build_or_update_index
from app.rag_pipeline import PatchContextRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PatchContext.API")

app = FastAPI(
    title="PatchContext API",
    description="Backend service for PatchContext RAG FastAPI history explorer.",
    version="1.0.0"
)

# Shared objects
db = DatabaseManager(settings.database_path)
rag = PatchContextRAG(db)

# Pydantic schemas
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    original_answer: str
    citations: Dict[str, Any]
    confidence_score: float
    latencies: Dict[str, float]

class SyncRequest(BaseModel):
    limit_commits: Optional[int] = 200
    limit_api: Optional[int] = 50

# In-memory progress state for background jobs
job_state = {"sync_running": False, "index_running": False, "status": "Idle"}

def background_sync(limit_commits: int, limit_api: int):
    """Worker task running GitHub extraction in the background."""
    global job_state
    job_state["sync_running"] = True
    job_state["status"] = "Syncing github data..."
    try:
        loader = GitHubLoader(db)
        loader.extract_commits(max_count=limit_commits)
        loader.extract_prs_graphql(limit=limit_api)
        loader.extract_issues_graphql(limit=limit_api)
        loader.export_data_to_json()
        job_state["status"] = "Sync completed successfully."
    except Exception as e:
        logger.error(f"Background sync failed: {e}", exc_info=True)
        job_state["status"] = f"Sync failed: {str(e)}"
    finally:
        job_state["sync_running"] = False

def background_reindex():
    """Worker task rebuilding vector index in the background."""
    global job_state
    job_state["index_running"] = True
    job_state["status"] = "Rebuilding vector index..."
    try:
        embeddings = get_embeddings()
        parser = DocumentParser()
        build_or_update_index(db, parser, embeddings)
        rag.refresh_retriever()
        job_state["status"] = "Reindexing completed successfully."
    except Exception as e:
        logger.error(f"Background reindexing failed: {e}", exc_info=True)
        job_state["status"] = f"Reindexing failed: {str(e)}"
    finally:
        job_state["index_running"] = False

@app.get("/")
def read_root():
    return {"message": "Welcome to PatchContext API. Use /docs for API documentation."}

@app.get("/stats")
def get_stats():
    """Returns total items stored in SQLite database."""
    try:
        return db.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_job_status():
    """Returns status of ongoing sync/indexing operations."""
    return job_state

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Submit a question to the PatchContext RAG pipeline."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        res = rag.run(request.question)
        return QueryResponse(
            question=res["question"],
            answer=res["answer"],
            original_answer=res["original_answer"],
            citations=res["citations"],
            confidence_score=res["confidence_score"],
            latencies=res["latencies"]
        )
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync")
def trigger_sync(request: SyncRequest, background_tasks: BackgroundTasks):
    """Trigger background synchronization with GitHub."""
    if job_state["sync_running"]:
        return {"message": "Synchronization is already running.", "status": job_state["status"]}
    
    background_tasks.add_task(
        background_sync, 
        request.limit_commits, 
        request.limit_api
    )
    return {"message": "Synchronization started in background."}

@app.post("/reindex")
def trigger_reindex(background_tasks: BackgroundTasks):
    """Trigger vector database rebuild in the background."""
    if job_state["index_running"]:
        return {"message": "Reindexing is already running.", "status": job_state["status"]}
        
    background_tasks.add_task(background_reindex)
    return {"message": "Reindexing started in background."}
