"""
tests/eval_suite.py

Evaluation pipeline for the Local Data Analyst Agent.

Runs a fixed set of queries against the agent and scores each on:
  - Tool accuracy:    were the expected tools actually called?
  - Response quality: does the response contain expected keywords?
  - Latency:          how long did the full agent loop take?

This is an integration eval — it REQUIRES Ollama to be running.
Run it with:

    python tests/eval_suite.py
    python tests/eval_suite.py --model mistral:7b-instruct-q4_0
    python tests/eval_suite.py --output eval_results.json

Exit code 0 = all evals passed (score >= threshold).
Exit code 1 = some evals failed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from orchestrator import DataAnalystAgent, SYSTEM_PROMPT
from memory.short_term import ShortTermMemory
from utils.logger import get_logger

log = get_logger(__name__)
console = Console()

# ── Eval case definition ──────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    query: str
    expected_tools: list[str]          # at least one of these must be called
    expected_keywords: list[str]       # at least one must appear in the response
    description: str = ""


@dataclass
class EvalResult:
    case_id: str
    query: str
    response: str
    tools_called: list[str]
    tool_score: float                  # 0.0 or 1.0
    keyword_score: float               # 0.0–1.0 (fraction of keywords found)
    latency_seconds: float
    passed: bool
    notes: str = ""


# ── Eval cases ────────────────────────────────────────────────────────────────

EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="E01",
        query="Find all CSV files in /tmp and list their names.",
        expected_tools=["file_search"],
        expected_keywords=["csv", "file", "/tmp"],
        description="Basic file discovery — must trigger file_search",
    ),
    EvalCase(
        id="E02",
        query="Compute the sum of [10, 20, 30, 40, 50] using Python.",
        expected_tools=["code_exec"],
        expected_keywords=["150", "sum"],
        description="Simple arithmetic via sandboxed code execution",
    ),
    EvalCase(
        id="E03",
        query="Calculate the mean and standard deviation of [5, 10, 15, 20, 25] in Python.",
        expected_tools=["code_exec"],
        expected_keywords=["15", "mean", "std"],
        description="Statistical computation — pandas or statistics module",
    ),
    EvalCase(
        id="E04",
        query="Scrape https://httpbin.org/html and tell me what you find.",
        expected_tools=["web_scrape"],
        expected_keywords=["html", "page", "content", "text", "web"],
        description="Web scraping integration",
    ),
    EvalCase(
        id="E05",
        query="Search my memory for anything about 'data analysis'.",
        expected_tools=["vector_search"],
        expected_keywords=["memory", "found", "result", "search", "analysis", "document", "empty"],
        description="RAG retrieval — must trigger vector_search",
    ),
    EvalCase(
        id="E06",
        query=(
            "Using Python, create a pandas DataFrame with columns 'name' and 'score', "
            "add three rows, and print the row with the highest score."
        ),
        expected_tools=["code_exec"],
        expected_keywords=["score", "max", "highest", "pandas", "dataframe"],
        description="Pandas DataFrame manipulation via code_exec",
    ),
    EvalCase(
        id="E07",
        query="Do you remember what we analysed earlier in this session?",
        expected_tools=["vector_search"],
        expected_keywords=["memory", "session", "earlier", "recall", "found", "previous", "nothing"],
        description="Memory recall — intent classifier should pre-route to vector_search",
    ),
    EvalCase(
        id="E08",
        query=(
            "Using Python, generate 20 random integers between 1 and 100, "
            "then print their sorted list and their median."
        ),
        expected_tools=["code_exec"],
        expected_keywords=["median", "sorted", "random"],
        description="Multi-step Python computation",
    ),
    EvalCase(
        id="E09",
        query="Find all .py files in /tmp recursively.",
        expected_tools=["file_search"],
        expected_keywords=[".py", "file", "python"],
        description="Recursive Python file discovery",
    ),
    EvalCase(
        id="E10",
        query=(
            "Using Python and DuckDB, create an in-memory table with two columns "
            "'product' and 'revenue', insert three rows, and print the total revenue."
        ),
        expected_tools=["code_exec"],
        expected_keywords=["revenue", "total", "duckdb", "sum"],
        description="DuckDB SQL via code_exec",
    ),
]

PASS_THRESHOLD = 0.6  # a case passes if combined score >= this

# ── Tool-call interception ────────────────────────────────────────────────────

class InstrumentedAgent(DataAnalystAgent):
    """
    Subclass that records every tool call made during run_query.
    We patch _run_tool to capture the names before delegating.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tool_calls_log: list[str] = []

    async def _run_tool(self, tool_name: str, arguments: dict) -> str:
        self.tool_calls_log.append(tool_name)
        return await super()._run_tool(tool_name, arguments)

    def reset_log(self) -> None:
        self.tool_calls_log = []
        # Fresh short-term memory for each eval case to avoid contamination
        self.st_memory = ShortTermMemory()
        self.st_memory.add("system", SYSTEM_PROMPT)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_tool(result: EvalResult, case: EvalCase) -> float:
    """1.0 if any expected tool was called, else 0.0."""
    expected = set(case.expected_tools)
    called = set(result.tools_called)
    return 1.0 if expected & called else 0.0


def score_keywords(result: EvalResult, case: EvalCase) -> float:
    """Fraction of expected keywords found in the response (case-insensitive)."""
    if not case.expected_keywords:
        return 1.0
    response_lower = result.response.lower()
    found = sum(1 for kw in case.expected_keywords if kw.lower() in response_lower)
    return round(found / len(case.expected_keywords), 2)


def combined_score(tool_score: float, keyword_score: float) -> float:
    return round((tool_score * 0.6) + (keyword_score * 0.4), 2)


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_eval(
    cases: list[EvalCase],
    model: str | None = None,
) -> list[EvalResult]:
    if model:
        cfg.OLLAMA_MODEL = model

    agent = InstrumentedAgent()
    results: list[EvalResult] = []

    for case in cases:
        console.print(f"\n[cyan]Running {case.id}:[/cyan] {case.description}")
        agent.reset_log()
        t0 = time.perf_counter()

        try:
            response = await agent.run_query(case.query)
        except Exception as exc:
            response = f"ERROR: {exc}"
            log.error(f"Eval {case.id} error: {exc}")

        latency = round(time.perf_counter() - t0, 3)
        tools_called = list(agent.tool_calls_log)

        partial = EvalResult(
            case_id=case.id,
            query=case.query,
            response=response,
            tools_called=tools_called,
            tool_score=0.0,
            keyword_score=0.0,
            latency_seconds=latency,
            passed=False,
        )
        partial.tool_score = score_tool(partial, case)
        partial.keyword_score = score_keywords(partial, case)
        score = combined_score(partial.tool_score, partial.keyword_score)
        partial.passed = score >= PASS_THRESHOLD

        status = "[green]PASS[/green]" if partial.passed else "[red]FAIL[/red]"
        console.print(
            f"  {status}  "
            f"tool={partial.tool_score:.1f}  "
            f"kw={partial.keyword_score:.2f}  "
            f"combined={score:.2f}  "
            f"latency={latency}s  "
            f"tools={tools_called}"
        )

        results.append(partial)

    await agent.shutdown()
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[EvalResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_latency = round(sum(r.latency_seconds for r in results) / total, 2)

    table = Table(title=f"\nEval Results  ({passed}/{total} passed)")
    table.add_column("ID", style="bold")
    table.add_column("Pass")
    table.add_column("Tool", justify="right")
    table.add_column("Keyword", justify="right")
    table.add_column("Combined", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Tools Called")

    for r in results:
        score = combined_score(r.tool_score, r.keyword_score)
        table.add_row(
            r.case_id,
            "[green]✓[/green]" if r.passed else "[red]✗[/red]",
            f"{r.tool_score:.1f}",
            f"{r.keyword_score:.2f}",
            f"{score:.2f}",
            f"{r.latency_seconds}s",
            ", ".join(r.tools_called) or "—",
        )

    console.print(table)
    console.print(f"\n[bold]Average latency:[/bold] {avg_latency}s")
    console.print(
        f"[bold]Overall:[/bold] "
        f"{'[green]PASSED[/green]' if passed == total else '[yellow]PARTIAL[/yellow]'}  "
        f"({passed}/{total} cases, threshold={PASS_THRESHOLD})"
    )


def save_results(results: list[EvalResult], output_path: str) -> None:
    data = {
        "model": cfg.OLLAMA_MODEL,
        "pass_threshold": PASS_THRESHOLD,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "avg_latency_seconds": round(
                sum(r.latency_seconds for r in results) / len(results), 2
            ),
        },
        "cases": [asdict(r) for r in results],
    }
    Path(output_path).write_text(json.dumps(data, indent=2))
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument("--model", help="Override Ollama model for this run")
    parser.add_argument("--output", help="Save JSON results to this file")
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Run only specific case IDs, e.g. --cases E01 E03",
    )
    args = parser.parse_args()

    cases = EVAL_CASES
    if args.cases:
        cases = [c for c in EVAL_CASES if c.id in args.cases]
        if not cases:
            console.print(f"[red]No matching cases for: {args.cases}[/red]")
            return 1

    console.print(f"[bold cyan]Eval suite[/bold cyan] — {len(cases)} case(s) — model: {cfg.OLLAMA_MODEL}")

    results = await run_eval(cases, model=args.model)
    print_report(results)

    if args.output:
        save_results(results, args.output)

    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
