"""
servers/web_server.py

MCP server exposing: web_scrape
Transport: stdio
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("web-server")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LocalDataAnalystAgent/1.0)"
    )
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_scrape",
            description="Fetch a URL and extract text content and HTML tables.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_text_chars": {"type": "integer", "default": 3000},
                },
                "required": ["url"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "web_scrape":
        raise ValueError(f"Unknown tool: {name}")

    url: str = arguments["url"]
    max_chars: int = arguments.get("max_text_chars", 3000)

    try:
        response = requests.get(url, headers=_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    soup = BeautifulSoup(response.text, "lxml")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract paragraph text
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if len(p) > 40)  # filter short noise
    text = text[:max_chars]

    # Extract tables (up to 3)
    tables: list[list[dict]] = []
    try:
        import pandas as pd

        for tbl in soup.find_all("table")[:3]:
            try:
                df = pd.read_html(str(tbl))[0]
                # Limit size
                tables.append(df.head(50).to_dict(orient="records"))
            except Exception:
                pass
    except ImportError:
        pass

    result = {
        "url": url,
        "text": text,
        "tables": tables,
        "table_count": len(tables),
    }
    return [TextContent(type="text", text=json.dumps(result, default=str))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
