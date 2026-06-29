"""
api.py

FastAPI REST API for the Local Data Analyst Agent.
Exposes the agent over HTTP so it can be used as a service,
not just an interactive CLI.

Endpoints:
  POST /chat            — single-turn query, returns agent response
  GET  /health          — liveness + Ollama status
  GET  /memory/search   — semantic search over long-term ChromaDB memory
  GET  /metrics         — latency and usage statistics
  DELETE /memory        — clear all long-term memory (dev/test utility)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config import cfg
from llm.ollama_client import OllamaClient
from mcp_client.dispatcher import MCPDispatcher
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from orchestrator import DataAnalystAgent, SYSTEM_PROMPT
from utils.logger import get_logger
from utils.metrics import metrics_store

log = get_logger(__name__)

# ── Shared agent instance (created once at startup) ───────────────────────────

_agent: DataAnalystAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start up the shared agent; shut it down on exit."""
    global _agent
    log.info("API startup: initialising agent…")
    _agent = DataAnalystAgent()
    yield
    log.info("API shutdown: cleaning up agent…")
    if _agent:
        await _agent.shutdown()


app = FastAPI(
    title="Local Data Analyst Agent API",
    description=(
        "REST interface for the fully local, privacy-preserving data analyst agent "
        "powered by Ollama, MCP tool calling, and ChromaDB RAG memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000, description="User query")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID to continue an existing conversation",
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
    duration_seconds: float
    tool_calls_made: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
    memory_chunks: int


class MemoryResult(BaseModel):
    text: str
    metadata: dict[str, Any]


class MetricsResponse(BaseModel):
    total_requests: int
    total_tool_calls: int
    avg_latency_seconds: float
    p95_latency_seconds: float
    errors: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Run a single query through the full agent loop:
    intent classification → tool calls → LLM synthesis → memory persistence.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    # If a session_id is supplied, swap the agent's short-term memory to that session
    if req.session_id:
        _agent.st_memory = ShortTermMemory(session_id=req.session_id)
        _agent.st_memory.add("system", SYSTEM_PROMPT)

    t0 = time.perf_counter()
    tool_calls_before = metrics_store.total_tool_calls

    try:
        response = await _agent.run_query(req.query)
    except Exception as exc:
        metrics_store.record_error()
        log.error(f"Chat endpoint error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    duration = round(time.perf_counter() - t0, 3)
    tool_calls_made = metrics_store.total_tool_calls - tool_calls_before
    metrics_store.record_request(latency=duration)

    return ChatResponse(
        response=response,
        session_id=_agent.st_memory.session_id,
        duration_seconds=duration,
        tool_calls_made=tool_calls_made,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness check — verifies Ollama is reachable."""
    llm = OllamaClient()
    ollama_ok = await llm.health_check()
    lt = LongTermMemory()
    try:
        chunk_count = lt.count()
    except Exception:
        chunk_count = -1

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_reachable=ollama_ok,
        model=cfg.OLLAMA_MODEL,
        memory_chunks=chunk_count,
    )


@app.get("/memory/search", response_model=list[MemoryResult])
async def memory_search(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(default=5, ge=1, le=20),
) -> list[MemoryResult]:
    """Semantic search over long-term ChromaDB memory."""
    lt = LongTermMemory()
    results = lt.search(query=q, top_k=top_k)
    return [MemoryResult(text=r["text"], metadata=r["metadata"]) for r in results]


@app.delete("/memory")
async def clear_memory() -> dict[str, str]:
    """
    Delete all long-term memory chunks from ChromaDB.
    Intended for development / testing — use with care in production.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(cfg.CHROMA_PERSIST_DIR))
    try:
        client.delete_collection(cfg.CHROMA_COLLECTION)
        return {"status": "cleared", "collection": cfg.CHROMA_COLLECTION}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Latency and usage statistics accumulated since last server start."""
    return MetricsResponse(
        total_requests=metrics_store.total_requests,
        total_tool_calls=metrics_store.total_tool_calls,
        avg_latency_seconds=metrics_store.avg_latency,
        p95_latency_seconds=metrics_store.p95_latency,
        errors=metrics_store.errors,
    )
