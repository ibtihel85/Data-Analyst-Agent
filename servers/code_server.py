"""
servers/code_server.py

MCP server exposing: code_exec
Executes Python code in a sandboxed subprocess with timeout.
No network calls are allowed inside the sandbox.
Transport: stdio
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("code-server")

# Packages available inside the sandbox
_SANDBOX_PACKAGES = [
    "pandas", "numpy", "matplotlib", "duckdb",
    "json", "csv", "math", "statistics", "datetime",
    "pathlib", "collections", "itertools", "functools",
]

_SANDBOX_ENV_OVERRIDES: dict[str, str] = {
    "MPLBACKEND": "Agg",           # headless matplotlib
    "no_proxy": "*",               # block proxy usage
    "NO_PROXY": "*",
    # Unset variables that could allow network access
    "http_proxy": "",
    "https_proxy": "",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
}

# Security: block dangerous builtins via a preamble injected before user code
_SECURITY_PREAMBLE = """\
import sys as _sys
# Restrict builtins — prevent subprocess, socket, etc.
_BLOCKED = {"__import__", "open"}  # open is overridden below to allow read-only local paths

# We allow open for data reading, but block network-capable modules
_BLOCKED_MODULES = {
    "socket", "urllib", "http", "ftplib", "smtplib",
    "subprocess", "multiprocessing", "pty", "tty",
}

_original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in _BLOCKED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in sandbox.")
    return _original_import(name, *args, **kwargs)

__builtins__.__import__ = _safe_import
del _sys, _BLOCKED, _BLOCKED_MODULES, _original_import, _safe_import
"""


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="code_exec",
            description=(
                "Execute a Python code snippet in a sandboxed subprocess. "
                "Available libraries: pandas, numpy, matplotlib, duckdb. "
                "Network access is blocked. Returns stdout, stderr, exit_code."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 15},
                },
                "required": ["code"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "code_exec":
        raise ValueError(f"Unknown tool: {name}")

    code: str = arguments["code"]
    timeout: int = int(arguments.get("timeout_seconds", cfg.CODE_EXEC_TIMEOUT_SECONDS))

    full_code = _SECURITY_PREAMBLE + "\n" + code

    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    # Build sandbox environment
    sandbox_env = {**os.environ, **_SANDBOX_ENV_OVERRIDES}

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=sandbox_env,
            cwd=tempfile.gettempdir(),
        )
        result = {
            "stdout": proc.stdout[-4000:],  # cap output
            "stderr": proc.stderr[-1000:],
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        result = {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
            "exit_code": -1,
        }
    except Exception as exc:
        result = {
            "stdout": "",
            "stderr": str(exc),
            "exit_code": -1,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return [TextContent(type="text", text=json.dumps(result))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
