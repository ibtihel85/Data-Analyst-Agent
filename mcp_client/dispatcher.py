"""
mcp_client/dispatcher.py

The MCP dispatcher manages stdio subprocesses for each MCP server,
sends tool calls via JSON-RPC 2.0, and returns results.

Each server is a Python subprocess communicating over stdin/stdout.
The protocol is a subset of JSON-RPC 2.0 as used by the MCP SDK.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

# Map tool name → server script path (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent

TOOL_SERVER_MAP: dict[str, Path] = {
    "file_search": PROJECT_ROOT / "servers" / "filesystem_server.py",
    "pdf_read": PROJECT_ROOT / "servers" / "pdf_server.py",
    "web_scrape": PROJECT_ROOT / "servers" / "web_server.py",
    "vector_search": PROJECT_ROOT / "servers" / "vector_server.py",
    "code_exec": PROJECT_ROOT / "servers" / "code_server.py",
}


@dataclass
class ServerProcess:
    script: Path
    proc: asyncio.subprocess.Process | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _msg_id: int = 0

    async def start(self) -> None:
        if self.proc and self.proc.returncode is None:
            return
        log.debug(f"Starting MCP server: {self.script.name}")
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self.script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        # Perform MCP initialize handshake
        await self._initialize()

    async def _send(self, message: dict) -> None:
        assert self.proc and self.proc.stdin
        data = json.dumps(message) + "\n"
        self.proc.stdin.write(data.encode())
        await self.proc.stdin.drain()

    async def _recv(self) -> dict:
        assert self.proc and self.proc.stdout
        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=30.0)
        if not line:
            raise EOFError("MCP server closed stdout")
        return json.loads(line.decode())

    async def _initialize(self) -> None:
        """Send MCP initialize request and consume the response."""
        self._msg_id += 1
        init_msg = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "local-analyst", "version": "1.0.0"},
            },
        }
        await self._send(init_msg)
        # Read until we get the initialize response
        for _ in range(10):
            try:
                msg = await self._recv()
                if msg.get("id") == self._msg_id:
                    break
            except asyncio.TimeoutError:
                break

        # Send initialized notification
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    async def call_tool(self, tool_name: str, arguments: dict, timeout: int = 30) -> Any:
        async with self._lock:
            await self.start()
            self._msg_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._msg_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            await self._send(request)

            # Read responses until we get the matching id
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError(f"Tool {tool_name} timed out")
                try:
                    msg = await asyncio.wait_for(self._recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError(f"Tool {tool_name} timed out")

                if msg.get("id") == self._msg_id:
                    if "error" in msg:
                        raise RuntimeError(f"MCP error: {msg['error']}")
                    return msg.get("result")
                # Ignore notifications / other messages

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self.proc.kill()


class MCPDispatcher:
    """
    Manages a pool of MCP server processes and dispatches tool calls.
    One process per server script (servers are shared across tools if on
    the same script — currently each tool has its own server).
    """

    def __init__(self) -> None:
        # Deduplicate: one ServerProcess per unique script path
        self._servers: dict[Path, ServerProcess] = {}
        for tool_name, script in TOOL_SERVER_MAP.items():
            if script not in self._servers:
                self._servers[script] = ServerProcess(script=script)

    def _server_for_tool(self, tool_name: str) -> ServerProcess:
        script = TOOL_SERVER_MAP.get(tool_name)
        if script is None:
            raise ValueError(f"No MCP server registered for tool: {tool_name}")
        return self._servers[script]

    async def call(
        self,
        tool_name: str,
        arguments: dict,
        timeout: int = 30,
    ) -> str:
        """
        Call a tool by name, returning the result as a JSON string.
        Raises on error or timeout.
        """
        server = self._server_for_tool(tool_name)
        log.info(f"[tool call] {tool_name}({json.dumps(arguments, default=str)[:120]})")

        result = await server.call_tool(tool_name, arguments, timeout=timeout)

        # MCP result format: {"content": [{"type": "text", "text": "..."}], ...}
        if isinstance(result, dict) and "content" in result:
            parts = [
                item.get("text", "")
                for item in result["content"]
                if item.get("type") == "text"
            ]
            text = "\n".join(parts)
        else:
            text = json.dumps(result, default=str)

        log.info(f"[tool result] {tool_name} → {text[:200]}")
        return text

    async def shutdown(self) -> None:
        for srv in self._servers.values():
            await srv.stop()
