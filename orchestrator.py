"""
orchestrator.py

Main entry point for the Local Data Analyst Agent.

Run:
    python orchestrator.py

The agent runs in an interactive CLI loop. Each user query triggers:
  1. Intent classification → optional pre-tool calls
  2. LLM turn with full tool manifest
  3. Tool execution loop (MCP dispatcher)
  4. Final response synthesis
  5. Memory persistence
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from config import cfg
from llm.ollama_client import OllamaClient
from mcp_client.dispatcher import MCPDispatcher
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from tools.schemas import TOOL_SCHEMAS
from utils.intent_classifier import classify
from utils.logger import get_logger

log = get_logger(__name__)
console = Console()

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise, local data analyst agent. You help users analyse data from
CSV files, PDFs, web pages, and databases using Python.

Guidelines:
- Always use the available tools to gather data before drawing conclusions.
- When writing code_exec scripts, prefer pandas for tabular data and
  matplotlib (Agg backend, save to /tmp/) for charts.
- Be concise and precise in your final answers.
- If a tool returns an error, ALWAYS retry with corrected parameters or switch
  to code_exec as a fallback. Never give up after one failure.
- NEVER invent, fabricate, or estimate data. Every number in your answer must
  come from an actual tool result in this conversation.
- If you cannot retrieve real data after retrying, say so explicitly.
- After receiving tool results, synthesise them into a clear answer.
- For optional integer parameters, always pass their default value, never null.
- If pre-fetched context is provided at the start of the message, use it directly
  to answer. Do not repeat the same tool call for data already retrieved.
- When a web_scrape result contains tables, extract and present the data from
  those tables rather than returning raw JSON.

Available tools: file_search, pdf_read, web_scrape, vector_search, code_exec.
"""


class DataAnalystAgent:
    def __init__(self) -> None:
        self.llm = OllamaClient()
        self.dispatcher = MCPDispatcher()
        self.lt_memory = LongTermMemory()
        self.st_memory = ShortTermMemory()

        # Seed the conversation with the system prompt
        self.st_memory.add("system", SYSTEM_PROMPT)

    # ── Tool execution ────────────────────────────────────────────────────

    async def _run_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a single tool via MCP dispatcher with timeout."""
        timeout = cfg.TOOL_TIMEOUT_SECONDS
        if tool_name == "code_exec":
            timeout = cfg.CODE_EXEC_TIMEOUT_SECONDS

        try:
            result = await asyncio.wait_for(
                self.dispatcher.call(tool_name, arguments, timeout=timeout),
                timeout=timeout + 5,
            )
            return result
        except asyncio.TimeoutError:
            return json.dumps({"error": f"Tool '{tool_name}' timed out after {timeout}s"})
        except Exception as exc:
            log.error(f"Tool error [{tool_name}]: {exc}")
            return json.dumps({"error": str(exc)})

    # ── Pre-routing ───────────────────────────────────────────────────────

    async def _pre_route(self, query: str) -> str:
        """
        Run intent-classified pre-tool calls and inject results as a
        system context message. Returns a context string to prepend.
        """
        pre_calls = classify(query)
        if not pre_calls:
            return ""

        context_parts: list[str] = []
        for pc in pre_calls:
            console.print(f"  [dim]↳ pre-routing → {pc.tool}[/dim]")
            result = await self._run_tool(pc.tool, pc.args)
            context_parts.append(f"[{pc.tool} result]\n{result}")

        if context_parts:
            return (
                "Pre-fetched context (use this data to answer the user — "
                "do NOT call the same tool again for the same URL):\n"
                + "\n\n".join(context_parts)
            )
        return ""

    # ── Agent loop ────────────────────────────────────────────────────────

    async def run_query(self, user_query: str) -> str:
        """
        Full agent loop for a single user query.
        Returns the final assistant response text.
        """
        # 1. Pre-routing
        console.print("[cyan]Classifying intent…[/cyan]")
        pre_context = await self._pre_route(user_query)

        # 2. Add user message (with optional pre-context injected)
        full_query = user_query
        if pre_context:
            full_query = f"{pre_context}\n\nUser question: {user_query}"

        self.st_memory.add("user", full_query)

        # 3. Tool execution loop
        iterations = 0
        final_response = ""

        while iterations < cfg.MAX_TOOL_ITERATIONS:
            iterations += 1
            console.print(f"[cyan]LLM turn {iterations}…[/cyan]")

            llm_resp = await self.llm.chat(
                messages=self.st_memory.messages(),
                tools=TOOL_SCHEMAS,
            )

            if not llm_resp.tool_calls:
                # No more tool calls — this is the final answer
                final_response = llm_resp.content
                self.st_memory.add("assistant", final_response)
                break

            # Record assistant message with tool_calls for context continuity
            raw_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in llm_resp.tool_calls
            ]
            self.st_memory.add_assistant_tool_call(
                content=llm_resp.content or "",
                tool_calls_raw=raw_tool_calls,
            )

            # Execute each tool call
            for tc in llm_resp.tool_calls:
                console.print(
                    f"  [yellow]⚙ tool:[/yellow] [bold]{tc.name}[/bold] "
                    f"[dim]{json.dumps(tc.arguments, default=str)[:100]}[/dim]"
                )
                result_text = await self._run_tool(tc.name, tc.arguments)
                self.st_memory.add_tool_result(
                    tool_call_id=tc.id,
                    content=result_text,
                )
                console.print(f"  [green]✓ result:[/green] [dim]{result_text[:150]}…[/dim]")
                # If the tool returned an error, nudge the LLM to retry or fallback
                if '"error"' in result_text or "validation error" in result_text.lower():
                    self.st_memory.add(
                        "user",
                        f"The tool call to '{tc.name}' failed with: {result_text}. "
                        "Please retry with corrected parameters — fix any null or missing "
                        "integer fields by using their default values. If the tool keeps "
                        "failing, use code_exec with requests and BeautifulSoup as a "
                        "fallback. Never invent or fabricate data.",
                    )

        else:
            # Hit iteration limit — ask LLM to summarise what it has
            self.st_memory.add(
                "user",
                "Please summarise your findings so far based on the tool results you have received.",
            )
            llm_resp = await self.llm.chat(messages=self.st_memory.messages())
            final_response = llm_resp.content
            self.st_memory.add("assistant", final_response)

        # 4. Persist memory
        self.st_memory.save()
        if final_response:
            self.lt_memory.store(
                text=final_response,
                session_id=self.st_memory.session_id,
                role="assistant",
                metadata={"query": user_query[:200]},
            )

        return final_response

    async def shutdown(self) -> None:
        await self.dispatcher.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Local Data Analyst Agent[/bold cyan]\n"
            f"Model: [green]{cfg.OLLAMA_MODEL}[/green]  |  "
            f"Ollama: [green]{cfg.OLLAMA_BASE_URL}[/green]\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.\n"
            "Type [bold]clear[/bold] to reset conversation memory.",
            title="🔬 Analyst Agent",
        )
    )

    agent = DataAnalystAgent()

    # Health check
    console.print("[dim]Checking Ollama connection…[/dim]")
    ok = await agent.llm.health_check()
    if not ok:
        console.print(
            "[red]⚠ Ollama is not reachable or model not found.[/red]\n"
            f"Start Ollama and run: [bold]ollama pull {cfg.OLLAMA_MODEL}[/bold]"
        )
        sys.exit(1)
    console.print(f"[green]✓ Ollama ready ({cfg.OLLAMA_MODEL})[/green]\n")

    try:
        while True:
            try:
                query = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit", "q"}:
                break
            if query.lower() == "clear":
                agent.st_memory.clear()
                agent.st_memory.add("system", SYSTEM_PROMPT)
                console.print("[yellow]Conversation cleared.[/yellow]")
                continue

            console.print()
            try:
                response = await agent.run_query(query)
                console.print("\n[bold green]Agent:[/bold green]")
                console.print(Markdown(response))
            except Exception as exc:
                log.error(f"Agent error: {exc}", exc_info=True)
                console.print(f"[red]Error: {exc}[/red]")

    finally:
        console.print("\n[dim]Shutting down…[/dim]")
        await agent.shutdown()
        console.print("[green]Bye![/green]")


if __name__ == "__main__":
    asyncio.run(main())
