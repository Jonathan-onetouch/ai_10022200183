# Student Name: jonatahn ahuche
# Student Index Number: 10022200183

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    """Centralized runtime configuration for the RAG system."""

    data_dir: str = "data"
    csv_path: str = "data/Ghana_Election_Result.csv"
    pdf_path: str = "data/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    cache_dir: str = "data/cache"
    index_path: str = "data/cache/vector.index"
    chunks_path: str = "data/cache/chunks.jsonl"
    logs_dir: str = "logs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.1-8b-instant"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    top_k: int = 5
    keyword_weight: float = 0.35
    vector_weight: float = 0.65
    max_context_chars: int = 5000
    temperature: float = 0.1


def get_llm_api_key() -> str:
    # Prefer Groq key for this project; fall back to OpenAI key.
    key = os.getenv("GROQ_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY (or OPENAI_API_KEY) is not set.")
    return key
