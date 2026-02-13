"""
Configuration settings for Cross-Border Compliance AI
All settings in one place for easy management
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM Settings
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0  # 0 for accuracy, higher for creativity
MAX_TOKENS = 2048

# Embedding Settings
EMBEDDING_MODEL = "nomic-embed-text"  # Ollama model
EMBEDDING_DIMENSION = 768

# Chunking Settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
SEPARATORS = ["\n\n", "\n", "; ", ". ", " ", ""]

# Retrieval Settings
RETRIEVAL_TOP_K = 10  # Number of chunks to retrieve
BM25_WEIGHT = 0.4    # Weight for keyword search
FAISS_WEIGHT = 0.6    # Weight for semantic search

# UI Settings
APP_TITLE = "🌍 Cross-Border Data Transfer Compliance Assistant"
APP_SUBTITLE = "Ask me about Kenya's cross-border data transfer requirements"

# Validate required settings
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")