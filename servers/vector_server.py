"""
servers/vector_server.py

MCP server exposing: vector_search
Uses ChromaDB + sentence-transformers all-MiniLM-L6-v2 (CPU-safe, ~80 MB).
Transport: stdio
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg

import chromadb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from sentence_transformers import SentenceTransformer

server = Server("vector-server")

# Lazy singletons — initialised once on first use
_embed_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _embed_model


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(cfg.CHROMA_PERSIST_DIR))
    return _chroma_client


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="vector_search",
            description=(
                "Semantic similarity search over the local ChromaDB vector store. "
                "Returns the most relevant document chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection": {"type": "string", "default": "analyst_memory"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "vector_search":
        raise ValueError(f"Unknown tool: {name}")

    query: str = arguments["query"]
    collection_name: str = arguments.get("collection", cfg.CHROMA_COLLECTION)
    top_k: int = arguments.get("top_k", 5)

    try:
        model = _get_embed_model()
        client = _get_chroma()
        collection = client.get_or_create_collection(collection_name)

        count = collection.count()
        if count == 0:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"documents": [], "metadatas": [], "note": "Collection is empty."}),
                )
            ]

        embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
        )

        payload = {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        }
        return [TextContent(type="text", text=json.dumps(payload, default=str))]

    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
