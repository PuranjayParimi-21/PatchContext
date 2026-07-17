import streamlit as st
import time
import os
import logging
from typing import Dict, Any, List
from app.config import settings
from app.database import DatabaseManager
from app.github_loader import GitHubLoader
from app.parser import DocumentParser
from app.embeddings import get_embeddings
from app.vector_store import build_or_update_index
from app.rag_pipeline import PatchContextRAG

# Setup page config
st.set_page_config(
    page_title="PatchContext - FastAPI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (sleek UI, card layouts, metric widgets, glassmorphism)
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #7d7d7d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics containers */
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Citations custom CSS badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 5px;
    }
    .badge-success {
        color: #fff;
        background-color: #28a745;
    }
    .badge-danger {
        color: #fff;
        background-color: #dc3545;
    }
    .badge-warning {
        color: #212529;
        background-color: #ffc107;
    }
    
    /* Citation Card list */
    .citation-list {
        background: #161b22;
        border-radius: 6px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-left: 4px solid #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PatchContext.Streamlit")

# Initialize database manager
@st.cache_resource
def get_db():
    return DatabaseManager(settings.database_path)

# Initialize RAG Pipeline
@st.cache_resource
def get_rag():
    db = get_db()
    return PatchContextRAG(db)

db = get_db()
rag_pipeline = get_rag()

# Sidebar: Configurations and statistics
st.sidebar.markdown("<h2 style='text-align: center; color: #FF4B4B;'>PatchContext Admin</h2>", unsafe_allow_html=True)

# 1. API Token Status Check
st.sidebar.subheader("Configuration Check")
openai_ok = bool(settings.openai_api_key)
github_ok = bool(settings.github_token)

col1, col2 = st.sidebar.columns(2)
with col1:
    if openai_ok:
        st.success("OpenAI Key OK")
    else:
        st.error("OpenAI Key Missing")
with col2:
    if github_ok:
        st.success("GitHub Token OK")
    else:
        st.warning("GitHub Token Missing")

# 2. Database Stats
st.sidebar.subheader("Repository Statistics")
try:
    stats = db.get_stats()
    st.sidebar.info(f"📁 Commits: {stats['commits']}")
    st.sidebar.info(f"🔄 Pull Requests: {stats['prs']}")
    st.sidebar.info(f"🐛 Issues: {stats['issues']}")
    st.sidebar.info(f"🔗 Relationships: {stats['relationships']}")
except Exception as e:
    st.sidebar.error(f"Error loading stats: {e}")

# 3. Actions: Sync and Index
st.sidebar.subheader("Data Sync Control")
limit_commits = st.sidebar.number_input("Commit sync limit", min_value=10, max_value=2000, value=200, step=50)
limit_api = st.sidebar.number_input("PR/Issue sync limit (per run)", min_value=10, max_value=500, value=50, step=10)

if st.sidebar.button("Run GitHub Sync (Incremental)", use_container_width=True):
    with st.spinner("Extracting commits, PRs, and issues..."):
        try:
            loader = GitHubLoader(db)
            st.info("Step 1: Extracting commits locally...")
            loader.extract_commits(max_count=limit_commits)
            
            st.info("Step 2: Syncing Pull Requests via GitHub API...")
            loader.extract_prs_graphql(limit=limit_api)
            
            st.info("Step 3: Syncing Issues via GitHub API...")
            loader.extract_issues_graphql(limit=limit_api)
            
            st.info("Step 4: Exporting cache JSON files...")
            loader.export_data_to_json()
            
            st.success("GitHub synchronization finished successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Sync failed: {e}")

if st.sidebar.button("Rebuild Vector Index", use_container_width=True):
    with st.spinner("Chunking and generating embeddings..."):
        try:
            embeddings = get_embeddings()
            parser = DocumentParser()
            st.info("Indexing new database items in FAISS...")
            vectorstore = build_or_update_index(db, parser, embeddings)
            if vectorstore:
                st.success("Vector index successfully updated!")
                # Force refresh retriever in the global pipeline
                rag_pipeline.refresh_retriever()
            else:
                st.warning("No new database items to index.")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

# Main panel
st.markdown("<h1 class='main-title'>PatchContext</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI assistant for FastAPI's development history, commits, pull requests, and issues.</p>", unsafe_allow_html=True)

# User Query Input
query = st.text_input("Ask a question about FastAPI's git history or decisions:", placeholder="e.g., Why was dependency injection introduced?")

# Search button triggering pipeline
if st.button("Search & Analyze History", type="primary") or query:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving evidence, cross-checking relations, and generating answer..."):
            # Execute pipeline
            result = rag_pipeline.run(query)
            
            # Answer Layout
            st.markdown("### Generated Answer")
            if result["answer"] == "I couldn't find sufficient evidence.":
                st.warning(result["answer"])
            else:
                st.markdown(result["answer"])
                
            # Performance Latency & Entailment Score Layout
            st.markdown("### Performance & Verification Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            latencies = result["latencies"]
            total_latency = latencies.get("total_response_latency", 0.0)
            ret_latency = latencies.get("total_retrieval_latency", 0.0)
            rerank_latency = latencies.get("rerank_latency", 0.0)
            llm_latency = latencies.get("llm_latency", 0.0)
            nli_score = result["confidence_score"]
            
            with col1:
                st.metric("Total Latency", f"{total_latency:.2f}s")
            with col2:
                st.metric("Retrieval Latency", f"{ret_latency:.2f}s")
            with col3:
                st.metric("Reranking Latency", f"{rerank_latency:.2f}s")
            with col4:
                st.metric("LLM Gen Latency", f"{llm_latency:.2f}s")
            with col5:
                st.metric("NLI Confidence", f"{nli_score*100:.1f}%")
                
            # Citation Verification Dashboard
            st.markdown("### Citation Verification")
            citations = result["citations"]
            
            has_citations = any(citations[category] for category in citations)
            
            if not has_citations:
                st.write("No citations found in the response.")
            else:
                # Group citations to render
                for sha, status in citations["commits"].items():
                    badge_style = "badge-success" if status["verified"] else "badge-danger"
                    status_text = "Verified: Exists in DB & Retrieved Context" if status["verified"] else "Hallucination Rejected"
                    if status["verified"]:
                        url = f"https://github.com/fastapi/fastapi/commit/{sha}"
                        link_html = f"<a href='{url}' target='_blank'><code>{sha[:7]}</code></a>"
                    else:
                        link_html = f"<code>{sha[:7]}</code>"
                    st.markdown(
                        f"<div class='citation-list'><span class='badge {badge_style}'>COMMIT</span> "
                        f"{link_html} &mdash; {status_text} (DB: {status['exists_db']}, Context: {status['in_context']})</div>",
                        unsafe_allow_html=True
                    )
                    
                for pr, status in citations["prs"].items():
                    badge_style = "badge-success" if status["verified"] else "badge-danger"
                    status_text = "Verified: Exists in DB & Retrieved Context" if status["verified"] else "Hallucination Rejected"
                    if status["verified"]:
                        url = f"https://github.com/fastapi/fastapi/pull/{pr}"
                        link_html = f"<a href='{url}' target='_blank'><code>#{pr}</code></a>"
                    else:
                        link_html = f"<code>#{pr}</code>"
                    st.markdown(
                        f"<div class='citation-list'><span class='badge {badge_style}'>PR</span> "
                        f"{link_html} &mdash; {status_text} (DB: {status['exists_db']}, Context: {status['in_context']})</div>",
                        unsafe_allow_html=True
                    )
                    
                for issue, status in citations["issues"].items():
                    badge_style = "badge-success" if status["verified"] else "badge-danger"
                    status_text = "Verified: Exists in DB & Retrieved Context" if status["verified"] else "Hallucination Rejected"
                    if status["verified"]:
                        url = f"https://github.com/fastapi/fastapi/issues/{issue}"
                        link_html = f"<a href='{url}' target='_blank'><code>#{issue}</code></a>"
                    else:
                        link_html = f"<code>#{issue}</code>"
                    st.markdown(
                        f"<div class='citation-list'><span class='badge {badge_style}'>ISSUE</span> "
                        f"{link_html} &mdash; {status_text} (DB: {status['exists_db']}, Context: {status['in_context']})</div>",
                        unsafe_allow_html=True
                    )
            
            # Retrieved Documents (Collapsible)
            with st.expander("Retrieved Context Chunks (Top Re-ranked Chunks)"):
                for idx, doc in enumerate(result["retrieved_docs"]):
                    m = doc.metadata
                    score = m.get("rerank_score", 0.0)
                    is_expanded = m.get("graph_expanded", False)
                    expanded_from = m.get("expanded_from", "")
                    
                    st.markdown(f"**Chunk #{idx+1} | Source: {m.get('type').upper()} (ID: {m.get('id')}) | Re-rank Score: {score:.4f}**")
                    if is_expanded:
                        st.caption(f"🕸️ *Graph-expanded context related to {expanded_from}*")
                    st.code(doc.page_content, language="markdown")
                    st.markdown("---")

            # Relationship Graph Connections
            with st.expander("SQLite Relationship Graph Visualization"):
                st.markdown("Relationships found in the database linking the retrieved resources:")
                rel_found = False
                seen_relations = set()
                for doc in result["retrieved_docs"]:
                    m = doc.metadata
                    item_type = m.get("type")
                    item_id = m.get("id")
                    
                    if item_type and item_id:
                        relations = db.get_related_items(item_type, item_id)
                        for target_type, target_id, rel_type in relations:
                            rel_key = tuple(sorted([f"{item_type}:{item_id}", f"{target_type}:{target_id}"])) + (rel_type,)
                            if rel_key not in seen_relations:
                                seen_relations.add(rel_key)
                                rel_found = True
                                emoji = " merges " if rel_type == "merges" else " references "
                                st.write(f"🕸️ **{item_type.upper()} #{item_id}** {emoji} **{target_type.upper()} #{target_id}** (Relation: *{rel_type}*)")
                if not rel_found:
                    st.write("No relationships found among the retrieved documents.")
