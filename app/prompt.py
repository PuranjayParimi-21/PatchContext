from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# Single HumanMessage prompt — avoids "system" role which many free OpenRouter models
# reject silently (returning empty output / "model output must contain text" error).
HUMAN_PROMPT = (
    "You are an AI assistant for FastAPI's development history.\n\n"
    "Use ONLY the context below to answer the question. "
    "Summarize in bullet points. Include key data (dates, authors, IDs).\n"
    "Cite sources as [Commit <SHA>], [PR <N>], or [Issue <N>].\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER:"
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
])
