"""
servers/pdf_server.py

MCP server exposing: pdf_read
Transport: stdio
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfplumber
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("pdf-server")


def _parse_page_range(pages_str: str | None, total: int) -> list[int]:
    """Parse '1-5' or '1,3,5' into zero-based page indices."""
    if not pages_str:
        return list(range(total))
    indices: list[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            s = max(0, int(start.strip()) - 1)
            e = min(total - 1, int(end.strip()) - 1)
            indices.extend(range(s, e + 1))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.append(idx)
    # Deduplicate while preserving order
    seen: set[int] = set()
    result: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="pdf_read",
            description="Extract text from a local PDF file, optionally for a page range.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pages": {"type": "string"},
                },
                "required": ["path"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "pdf_read":
        raise ValueError(f"Unknown tool: {name}")

    path = Path(arguments["path"]).expanduser().resolve()
    if not path.exists():
        return [TextContent(type="text", text=json.dumps({"error": f"File not found: {path}"}))]
    if not path.suffix.lower() == ".pdf":
        return [TextContent(type="text", text=json.dumps({"error": "Not a PDF file"}))]

    pages_str: str | None = arguments.get("pages")
    extracted: list[str] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            total = len(pdf.pages)
            page_indices = _parse_page_range(pages_str, total)
            for i in page_indices:
                text = pdf.pages[i].extract_text() or ""
                if text.strip():
                    extracted.append(f"--- Page {i + 1} ---\n{text}")
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    result = {
        "path": str(path),
        "pages_read": len(extracted),
        "text": "\n\n".join(extracted),
    }
    return [TextContent(type="text", text=json.dumps(result))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
