"""
servers/filesystem_server.py

MCP server exposing: file_search
Transport: stdio
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("filesystem-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="file_search",
            description="Find files matching a glob pattern in a local directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "pattern": {"type": "string"},
                    "recursive": {"type": "boolean", "default": True},
                },
                "required": ["directory", "pattern"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "file_search":
        raise ValueError(f"Unknown tool: {name}")

    directory = str(Path(arguments["directory"]).expanduser().resolve())
    pattern = arguments["pattern"]
    recursive = arguments.get("recursive", True)

    if recursive:
        search_pattern = os.path.join(directory, "**", pattern)
    else:
        search_pattern = os.path.join(directory, pattern)

    matches = glob.glob(search_pattern, recursive=recursive)

    results = []
    for f in matches[:50]:  # cap at 50 results
        try:
            size_kb = round(os.path.getsize(f) / 1024, 1)
        except OSError:
            size_kb = 0.0
        results.append({"path": f, "size_kb": size_kb})

    return [TextContent(type="text", text=json.dumps(results, indent=2))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
