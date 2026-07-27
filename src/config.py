"""Central configuration for the RAG pipeline."""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "assignment_rag_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_TARGET_WORDS = 500
CHUNK_OVERLAP_WORDS = 100
TOP_K = 4
MAX_RETRIEVAL_DISTANCE = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "0.80"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
