import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load Streamlit Cloud secrets into environment variables for Pydantic Settings
try:
    import streamlit as st
    for k in st.secrets.keys():
        val = st.secrets[k]
        if isinstance(val, (str, int, float, bool)):
            os.environ[k] = str(val)
            os.environ[k.upper()] = str(val)
except Exception:
    pass

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    github_token: Optional[str] = Field(default=None, validation_alias="GITHUB_TOKEN")
    
    database_path: str = Field(default="data/metadata.db", validation_alias="DATABASE_PATH")
    vectorstore_path: str = Field(default="vectorstore", validation_alias="VECTORSTORE_PATH")
    
    github_repository: str = Field(default="fastapi/fastapi", validation_alias="GITHUB_REPOSITORY")
    local_repo_path: str = Field(default="data/fastapi_repo", validation_alias="LOCAL_REPO_PATH")
    
    nli_model_name: str = Field(default="facebook/bart-large-mnli", validation_alias="NLI_MODEL_NAME")
    enable_nli_guard: bool = Field(default=False, validation_alias="ENABLE_NLI_GUARD")
    nli_entailment_threshold: float = Field(default=0.3, validation_alias="NLI_ENTAILMENT_THRESHOLD")
    embedding_provider: str = Field(default="local", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL")
    llm_provider: str = Field(default="openrouter", validation_alias="LLM_PROVIDER")
    model: str = Field(default="meta-llama/llama-3.2-3b-instruct:free", validation_alias="MODEL")
    openrouter_api_key: Optional[str] = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="meta-llama/llama-3.2-3b-instruct:free", validation_alias="OPENROUTER_MODEL")

    def __init__(self, **values):
        super().__init__(**values)
        if not self.openai_api_key or self.openai_api_key.strip() == "":
            self.openai_api_key = "missing-api-key"

# Instantiate a global settings object
settings = Settings()

# Ensure parent directories for database and repository exist
db_dir = os.path.dirname(settings.database_path)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)
    
repo_dir = os.path.dirname(settings.local_repo_path)
if repo_dir and not os.path.exists(repo_dir):
    os.makedirs(repo_dir, exist_ok=True)
