"""
utils/intent_classifier.py

Rule-based intent classifier for pre-routing queries before the LLM turn.
Returns a list of (tool_name, pre_args) tuples to pre-call before the main
LLM invocation, giving the LLM richer context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PreCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


# Patterns
_URL_RE = re.compile(r"https?://[^\s\"']+")
_PDF_RE = re.compile(r"[\w./~-]+\.pdf\b", re.IGNORECASE)
_CSV_RE = re.compile(r"[\w./~-]+\.csv\b", re.IGNORECASE)
_PATH_KEYWORDS = re.compile(
    r"\b(find|list|search|locate|show)\b.*\bfile", re.IGNORECASE
)
_MEMORY_KEYWORDS = re.compile(
    r"\b(remember|recall|earlier|previously|last time|before)\b", re.IGNORECASE
)
_ANALYSIS_KEYWORDS = re.compile(
    r"\b(analyse|analyze|correlation|trend|chart|plot|graph|statistic|summarize|summary)\b",
    re.IGNORECASE,
)


def classify(query: str) -> list[PreCall]:
    """
    Return a list of PreCall objects describing tools to call before the LLM
    turn. The orchestrator feeds these results into the context window.
    """
    calls: list[PreCall] = []

    # URL detected → web_scrape
    urls = _URL_RE.findall(query)
    for url in urls[:1]:  # limit to first URL to avoid noise
        calls.append(PreCall(tool="web_scrape", args={"url": url}))

    # PDF path detected → pdf_read
    pdfs = _PDF_RE.findall(query)
    for pdf_path in pdfs[:1]:
        expanded = str(Path(pdf_path).expanduser())
        calls.append(PreCall(tool="pdf_read", args={"path": expanded}))

    # CSV path detected → file_search to confirm existence
    csvs = _CSV_RE.findall(query)
    for csv_path in csvs[:1]:
        p = Path(csv_path).expanduser()
        calls.append(
            PreCall(
                tool="file_search",
                args={
                    "directory": str(p.parent),
                    "pattern": p.name,
                    "recursive": False,
                },
            )
        )

    # Explicit "find files" intent → file_search in home directory
    if _PATH_KEYWORDS.search(query) and not calls:
        calls.append(
            PreCall(
                tool="file_search",
                args={
                    "directory": str(Path.home()),
                    "pattern": "*.csv",
                    "recursive": True,
                },
            )
        )

    # Memory-referencing query → vector_search for past context
    if _MEMORY_KEYWORDS.search(query):
        calls.append(PreCall(tool="vector_search", args={"query": query, "top_k": 3}))

    return calls
