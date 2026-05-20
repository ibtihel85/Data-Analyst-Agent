"""
tools/schemas.py

Canonical JSON schemas for every tool exposed to the LLM via Ollama's
tools array.  These must stay in sync with the server implementations.
"""
from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": (
                "Search for files matching a glob pattern in a local directory. "
                "Returns a list of matching file paths with their sizes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Absolute path to the directory to search.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '*.csv' or '**/*.pdf'.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to search subdirectories recursively.",
                    },
                },
                "required": ["directory", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdf_read",
            "description": (
                "Extract text from a local PDF file. "
                "Optionally specify a page range such as '1-5'. "
                "Returns the extracted text content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the PDF file.",
                    },
                    "pages": {
                        "type": "string",
                        "description": (
                            "Page range to read, e.g. '1-5' or '2,4,6'. "
                            "Omit to read all pages."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": (
                "Fetch a web page URL and extract its text content and any HTML tables. "
                "Returns cleaned paragraph text and tables as JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to scrape (http or https).",
                    },
                    "max_text_chars": {
                        "type": "integer",
                        "default": 3000,
                        "description": "Maximum characters of text to return.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "Semantic similarity search over documents previously indexed in the "
                "local ChromaDB vector store. Use to retrieve relevant context from "
                "past analyses or ingested documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "collection": {
                        "type": "string",
                        "default": "analyst_memory",
                        "description": "ChromaDB collection name.",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_exec",
            "description": (
                "Execute a Python code snippet in a sandboxed subprocess. "
                "Use for data analysis with pandas, matplotlib charts (saved to /tmp), "
                "statistical calculations, or any computation. "
                "Captures stdout/stderr. Returns exit code and output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "default": 15,
                        "description": "Maximum execution time in seconds.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]
