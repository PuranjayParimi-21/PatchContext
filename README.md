# PatchContext 🤖

PatchContext is a production-quality Retrieval-Augmented Generation (RAG) application designed to serve as an AI-powered assistant for software engineers exploring the development history, design decisions, and PR integrations of the **FastAPI** GitHub repository.

By retrieving evidence from local Git commit histories, Pull Requests, and Issue discussions, PatchContext constructs accurate responses with strict citation validation to avoid hallucinations.

---

## Key Features

1. **GitHub GraphQL Data Extraction:** Extracts commits (via GitPython), Pull Requests, and Issues, supporting complete paginated downloads, checkpoint resume, and incremental caching.
2. **Relationship Graph Storage:** Stores metadata in SQLite and links entities (e.g., `Issue` $\rightarrow$ `PR` $\rightarrow$ `Merge Commit`) to enable relationship-aware context expansion.
3. **Hybrid Search (BM25 + Vector MMR):** Combines exact keyword matching (BM25) with semantic embeddings using `text-embedding-ada-002` in FAISS, leveraging LangChain MMR (`search_type="mmr"`) with optimized parameters.
4. **Cross-Encoder Re-ranking:** Re-ranks the combined context using `cross-encoder/ms-marco-MiniLM-L-6-v2` to deliver the top 5 highly relevant text fragments to the LLM.
5. **Incremental Vector Indexing:** Employs tracking status flags in SQLite to embed and index only newly extracted data, with automated database state resetting to facilitate full FAISS index rebuilding when needed.
6. **Hallucination Guard (BART NLI):** Performs zero-shot Natural Language Inference (`facebook/bart-large-mnli`) comparing every generated claim against retrieved context. Filters out unsupported sentences below a `0.5` threshold or rejects the answer, enabled by default. Regex citation checks act as an additional validation layer.
7. **Premium Streamlit Dashboard:** Displays live sync progress, repository sync statistics, latency metrics (retrieval, reranking, LLM, verification), confidence scores, and expandable source segments with clickable GitHub citations.

---

## Project Structure

```
PatchContext/
├── app/
│   ├── __init__.py
│   ├── api.py            # FastAPI endpoints (Optional backend wrapper)
│   ├── config.py         # Configs (Pydantic settings loading from .env)
│   ├── database.py       # SQLite connection manager & relationship graph table
│   ├── github_loader.py  # GitPython & GitHub GraphQL loader with paginated resume
│   ├── parser.py         # Formats raw SQL records into LangChain Documents & chunking
│   ├── embeddings.py     # OpenAI Embeddings setup (text-embedding-ada-002)
│   ├── vector_store.py   # FAISS load, save, and incremental indexing logic
│   ├── retriever.py      # BM25 + Vector search, Graph Expansion, and Re-ranking
│   ├── prompt.py         # System prompting with strict citation rules
│   ├── rag_pipeline.py   # Full QA RAG flow pipeline
│   ├── verifier.py       # Citation verifier & BART NLI entailment checker
│   └── evaluator.py      # Evaluates pipeline using RAGAs metrics
├── data/
│   ├── commits.json      # JSON cache of commits
│   ├── prs.json          # JSON cache of PRs
│   ├── issues.json       # JSON cache of Issues
│   └── metadata.db       # SQLite Database containing metadata & relationship graph
├── tests/                # Complete PyTest Suite
│   ├── test_database.py
│   ├── test_github_loader.py
│   ├── test_parser.py
│   ├── test_vector_store.py
│   └── test_verifier.py
├── vectorstore/          # FAISS persisted index files
├── benchmark/
│   ├── questions.json    # 50 manually written engineering questions
│   └── answers.json      # RAGAs evaluation output report
├── streamlit_app.py      # Streamlit user interface
├── requirements.txt      # Python dependencies list
├── Dockerfile            # Container configuration
├── .env                  # Private configurations
└── .github/
    └── workflows/
        └── deploy.yml    # CI/CD test and container build workflow
```

---

## Setup Instructions

### Prerequisites
* Python 3.12+ installed
* Git installed

### 1. Clone the project and configure environment
Copy `.env` from template and fill in your keys:
```bash
cp .env.example .env
```
Ensure you set the following in `.env`:
* `OPENAI_API_KEY`: Your OpenAI API credentials.
* `GITHUB_TOKEN`: A GitHub Personal Access Token (PAT) to perform GraphQL requests.
* `LLM_PROVIDER`: LLM provider name (defaults to `openai`).
* `MODEL`: LLM model name (defaults to `gpt-4o-mini`).
* `EMBEDDING_MODEL`: Embedding model name (defaults to `text-embedding-ada-002`).
* `ENABLE_NLI_GUARD`: Enables/disables the BART NLI hallucination check (defaults to `true`).

### 2. Install dependencies
Initialize virtualenv and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Application

### 1. Sync Repository & Build Index
You can sync repository data and build the search index directly inside the Streamlit Admin panel, or execute it via python script.
To perform data synchronization and build vector indices for the first time:

Start the Streamlit interface:
```bash
streamlit run streamlit_app.py
```
1. Click **Run GitHub Sync (Incremental)** in the sidebar to download commits, PRs, and issues.
2. Click **Rebuild Vector Index** to chunk records, embed them, and save the index.

### 2. Ask Questions
Once the index is built, type your question in the search input field and click **Search & Analyze History**.

### 3. Run Unit Tests
To verify all core modules operate properly offline:
```bash
pytest tests/
```

### 4. Run RAGAs Evaluation Benchmarking
To run the automated RAGAs metrics evaluation on your index:
```bash
# Evaluate on 5 sample questions
python -m app.evaluator --limit 5

# Evaluate on all 50 questions
python -m app.evaluator --limit 50
```
This evaluates the pipeline metrics (`Faithfulness`, `Answer Relevancy`, `Context Recall`, `Context Precision`, and `Answer Correctness`) and exports the report to `benchmark/answers.json`.

---

## Deployment Instructions

### Docker deployment
You can build and deploy the container locally or in the cloud:

```bash
# Build the container
docker build -t patchcontext:latest .

# Run the container
docker run -p 8501:8501 --env-file .env patchcontext:latest
```
Visit `http://localhost:8501` to use the application.
