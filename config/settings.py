"""
config/settings.py
==================
Centralized project configuration.
Loads values from .env and exposes constants + helpers.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# -- Load environment variables ------------------------------
load_dotenv()

# -- Base directories ----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "papers"
OUTPUT_DIR = PROJECT_ROOT / "output"

# -- Ollama (Local LLM) --------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# -- Embeddings -----------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# -- RAG / chunking ------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "5"))

# -- ChromaDB -------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "research_papers")

# -- Orchestrator ----------------------------------------------
MAX_RETRY_COUNT = 2  # Maximum retries from Validator -> Writer


def get_llm():
    """Returns an instance of ChatOllama configured from the environment."""
    from langchain_community.chat_models import ChatOllama

    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,  # Low temperature for consistent responses
    )


def get_embed_model():
    """Returns the HuggingFace embedding model (local, free)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
