"""
config.py — centralised configuration loaded from .env
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


class Config:
    # ── Ollama ────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    # ── Agent loop ────────────────────────────────────────────────────────
    MAX_TOOL_ITERATIONS: int = int(os.getenv("MAX_TOOL_ITERATIONS", "10"))
    TOOL_TIMEOUT_SECONDS: int = int(os.getenv("TOOL_TIMEOUT_SECONDS", "15"))
    CODE_EXEC_TIMEOUT_SECONDS: int = int(os.getenv("CODE_EXEC_TIMEOUT_SECONDS", "15"))

    # ── Memory ────────────────────────────────────────────────────────────
    SESSION_MEMORY_DIR: Path = Path(os.getenv("SESSION_MEMORY_DIR", "./data/sessions"))
    CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_store"))
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "analyst_memory")
    SHORT_TERM_MEMORY_SIZE: int = int(os.getenv("SHORT_TERM_MEMORY_SIZE", "20"))

    # ── Embeddings ────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # ~80 MB, CPU-safe

    # ── API server ────────────────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    
    @classmethod
    def ensure_dirs(cls) -> None:
        cls.SESSION_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg.ensure_dirs()
