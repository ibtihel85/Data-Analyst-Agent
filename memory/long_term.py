"""
memory/long_term.py

Long-term memory using ChromaDB + sentence-transformers (all-MiniLM-L6-v2).
Stores assistant responses chunked and embedded for future semantic retrieval.
CPU-safe: no GPU required.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_CHUNK_SIZE = 400       # characters per chunk
_CHUNK_OVERLAP = 80     # character overlap between chunks


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return [c for c in chunks if c.strip()]


class LongTermMemory:
    """
    Wraps ChromaDB for persistent semantic memory.
    Lazily initialises the embedding model on first use to avoid
    slowing down startup.
    """

    def __init__(self) -> None:
        self._client: chromadb.PersistentClient | None = None
        self._collection: Any = None
        self._model: SentenceTransformer | None = None

    def _init(self) -> None:
        if self._client is not None:
            return
        log.debug("Initialising ChromaDB + embedding model (first use)…")
        try:
            from pathlib import Path
            db_path = Path(cfg.CHROMA_PERSIST_DIR).expanduser().resolve()
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False),
            )
            log.debug(f"ChromaDB PersistentClient initialised at {db_path}")
        except Exception as exc:
            log.error(f"ChromaDB PersistentClient failed ({exc}), falling back to in-memory client.")
            self._client = chromadb.Client(Settings(anonymized_telemetry=False))

        self._collection = self._client.get_or_create_collection(
            cfg.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._model = SentenceTransformer(cfg.EMBEDDING_MODEL)
        log.debug(f"ChromaDB ready. Collection '{cfg.CHROMA_COLLECTION}' has {self._collection.count()} docs.")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        assert self._model is not None
        return self._model.encode(texts, show_progress_bar=False).tolist()

    # ── Write ─────────────────────────────────────────────────────────────

    def store(
        self,
        text: str,
        session_id: str,
        role: str = "assistant",
        metadata: dict | None = None,
    ) -> int:
        """Chunk text, embed, and upsert into ChromaDB. Returns chunk count."""
        if not text.strip():
            return 0
        self._init()
        assert self._collection is not None

        chunks = _chunk_text(text)
        embeddings = self._embed(chunks)
        now = datetime.utcnow().isoformat()

        ids: list[str] = []
        metas: list[dict] = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{session_id}:{now}:{i}:{chunk}".encode()).hexdigest()
            ids.append(chunk_id)
            meta = {
                "session_id": session_id,
                "role": role,
                "chunk_index": i,
                "timestamp": now,
            }
            if metadata:
                meta.update(metadata)
            metas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metas,
        )
        log.debug(f"Stored {len(chunks)} chunks in long-term memory.")
        return len(chunks)

    # ── Read ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        session_filter: str | None = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of dicts with 'text' and 'metadata'.
        """
        self._init()
        assert self._collection is not None

        count = self._collection.count()
        if count == 0:
            return []

        embedding = self._embed([query])[0]
        where = {"session_id": session_filter} if session_filter else None

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            where=where,
        )

        output: list[dict] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            output.append({"text": doc, "metadata": meta})
        return output

    def count(self) -> int:
        self._init()
        assert self._collection is not None
        return self._collection.count()
