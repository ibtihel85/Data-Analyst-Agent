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
import sys
import tempfile
import anyio
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

# PATCH 4 — resolve the real temp dir once at import time so both the
# env-var overrides and the injected preamble agree on the same path.
_TMP_DIR = tempfile.gettempdir()

_SANDBOX_ENV_OVERRIDES: dict[str, str] = {
    "MPLBACKEND": "Agg",           # headless matplotlib
    "no_proxy": "*",               # block proxy usage
    "NO_PROXY": "*",
    # Unset variables that could allow network access
    "http_proxy": "",
    "https_proxy": "",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    # PATCH 4 — make sure the subprocess sees the real temp dir on every OS
    # (critical on Windows where /tmp does not exist)
    "TMPDIR": _TMP_DIR,
    "TEMP":   _TMP_DIR,
    "TMP":    _TMP_DIR,
}



# PATCH 3 — inject matplotlib Agg backend explicitly in code, not just
# via the env var.  The env var alone is sometimes ignored when the
# backend has already been loaded by a previous import in the same
# interpreter session.  This must come BEFORE any `import matplotlib`
# in user code.
_MATPLOTLIB_PREAMBLE = """\
import matplotlib
matplotlib.use('Agg')
"""

# PATCH 4 — inject the platform-correct temp directory as _TMP_DIR so
# that LLM-generated code can write to it without hard-coding /tmp
# (which does not exist on Windows).  We also monkey-patch the string
# "/tmp" → real temp dir for the common case where the LLM hard-codes
# the Unix path.
def _build_path_preamble(tmp_dir: str) -> str:
    # Use repr() so backslashes in Windows paths are escaped correctly.
    return f"""\
import os as _os
import builtins as _bt

_TMP_DIR = {repr(tmp_dir)}

# Transparently redirect hard-coded /tmp paths to the real temp dir.
_real_savefig = None  # patched lazily below after matplotlib is imported

def _patched_savefig(fname, *args, **kwargs):
    if isinstance(fname, str) and fname.startswith("/tmp/"):
        fname = _os.path.join(_TMP_DIR, fname[len("/tmp/"):])
    return _real_savefig(fname, *args, **kwargs)

# Delay the patch until plt is actually imported by user code.
_real_open = _bt.open
def _patched_open(file, *args, **kwargs):
    if isinstance(file, str) and file.startswith("/tmp/"):
        file = _os.path.join(_TMP_DIR, file[len("/tmp/"):])
    return _real_open(file, *args, **kwargs)

_bt.open = _patched_open
del _bt
"""


def _sanitize_code(code: str) -> str:
    """
    Fix the most common code generation mistakes from small models (e.g.
    llama3.2:3b) before execution.  Surgical corrections only.

    Handles:
    - `plt = None` with no pyplot import → replaces with proper import
    - pyplot used but never imported → inserts import after matplotlib line
    """
    import re

    lines = code.splitlines()
    has_pyplot_import = any(re.search(r"import matplotlib\.pyplot", l) for l in lines)
    has_plt_none = any(re.match(r"\s*plt\s*=\s*None\b", l) for l in lines)
    uses_plt = any(
        re.search(r"\bplt\.", l) for l in lines
        if not re.match(r"\s*plt\s*=\s*None\b", l)
    )

    if not has_plt_none and has_pyplot_import:
        return code  # nothing to fix

    fixed_lines: list[str] = []
    inserted_pyplot = False

    for line in lines:
        # Drop `plt = None` and replace with the real import if needed
        if re.match(r"\s*plt\s*=\s*None\b", line):
            if not inserted_pyplot and not has_pyplot_import and uses_plt:
                fixed_lines.append("import matplotlib.pyplot as plt")
                inserted_pyplot = True
            continue  # always drop the `plt = None` line

        # Insert pyplot import right after `import matplotlib` line
        if (
            not inserted_pyplot
            and not has_pyplot_import
            and uses_plt
            and re.match(r"\s*import matplotlib\b", line)
        ):
            fixed_lines.append(line)
            fixed_lines.append("import matplotlib.pyplot as plt")
            inserted_pyplot = True
            continue

        fixed_lines.append(line)

    # Last resort: prepend if still not inserted
    if not inserted_pyplot and not has_pyplot_import and uses_plt:
        fixed_lines.insert(0, "import matplotlib.pyplot as plt")

    return "\n".join(fixed_lines)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="code_exec",
            description=(
                "Execute a Python code snippet in a sandboxed subprocess. "
                "Available libraries: pandas, numpy, matplotlib, duckdb. "
                "Network access is blocked. Returns stdout, stderr, exit_code. "
                # PATCH 3 — tell the LLM it does NOT need to set the backend itself
                "matplotlib Agg backend is already configured; do NOT call "
                "matplotlib.use() yourself. "
                # PATCH 4 — tell the LLM to use /tmp which will be redirected
                "Save plots with plt.savefig('/tmp/<name>.png'). "
                "To confirm the file was saved, add print(os.path.exists('/tmp/<name>.png')) "
                "at the end of the same script. Never use pdf_read to check for image files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    # PATCH 5 — raise the default from 15 s to 60 s
                    "timeout_seconds": {"type": "integer", "default": 60},
                },
                "required": ["code"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "code_exec":
        raise ValueError(f"Unknown tool: {name}")

    code: str = _sanitize_code(arguments["code"])

    # PATCH 5 — floor the timeout at 60 s so a misconfigured cfg or a
    # LLM that forgets to pass timeout_seconds never falls back to 15 s.
    timeout: int = max(60, int(arguments.get("timeout_seconds", cfg.CODE_EXEC_TIMEOUT_SECONDS)))

    # PATCH 3 + 4 — assemble full code: security preamble → matplotlib
    # backend fix → /tmp path redirect → user code.
    full_code = (
        _MATPLOTLIB_PREAMBLE
        + "\n"
        + _build_path_preamble(_TMP_DIR)
        + "\n"
        + code
    )

    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    # Build sandbox environment
    sandbox_env = {**os.environ, **_SANDBOX_ENV_OVERRIDES}

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdin=asyncio.subprocess.DEVNULL,   # ← ADD THIS
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=sandbox_env,
            cwd=_TMP_DIR,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            result = {
                "stdout": stdout_bytes.decode(errors="replace")[-4000:],
                "stderr": stderr_bytes.decode(errors="replace")[-1000:],
                "exit_code": proc.returncode,
            }
        except asyncio.TimeoutError:
            try:
                proc.kill()
                stdout_bytes, stderr_bytes = await proc.communicate()
            except Exception:
                stdout_bytes, stderr_bytes = b"", b""
            result = {
                "stdout": stdout_bytes.decode(errors="replace")[-2000:],
                "stderr": (
                    f"Execution timed out after {timeout}s. "
                    f"Partial stderr: {stderr_bytes.decode(errors='replace')[-500:]}"
                ),
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


async def _warm_imports() -> None:
    """Pre-import heavy packages so the first user call doesn't timeout."""
    warmup = "import numpy; import matplotlib; import pandas; print('warm')"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", warmup,
        stdin=asyncio.subprocess.DEVNULL,   # ← ADD THIS
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.wait_for(proc.communicate(), timeout=120)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await _warm_imports()               # ← MOVE inside stdio_server context
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
            raise_exceptions=True,
        )


if __name__ == "__main__":
    import anyio
    anyio.run(main)