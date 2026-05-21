"""
tests/test_tools.py

Unit tests for individual tool implementations.
Run with: pytest tests/ -v

These tests do NOT require Ollama — they test the tool logic directly.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── file_search ───────────────────────────────────────────────────────────────

class TestFileSearch:
    def test_finds_existing_file(self, tmp_path):
        (tmp_path / "test.csv").write_text("a,b\n1,2")
        (tmp_path / "other.txt").write_text("hello")

        from servers.filesystem_server import call_tool

        result = run(call_tool("file_search", {"directory": str(tmp_path), "pattern": "*.csv"}))
        data = json.loads(result[0].text)
        assert len(data) == 1
        assert data[0]["path"].endswith("test.csv")

    def test_returns_empty_for_no_match(self, tmp_path):
        from servers.filesystem_server import call_tool

        result = run(call_tool("file_search", {"directory": str(tmp_path), "pattern": "*.pdf"}))
        data = json.loads(result[0].text)
        assert data == []

    def test_recursive_search(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.csv").write_text("x,y")

        from servers.filesystem_server import call_tool

        result = run(
            call_tool("file_search", {"directory": str(tmp_path), "pattern": "*.csv", "recursive": True})
        )
        data = json.loads(result[0].text)
        assert any("deep.csv" in f["path"] for f in data)


# ── pdf_read ──────────────────────────────────────────────────────────────────

class TestPdfRead:
    def test_nonexistent_file(self):
        from servers.pdf_server import call_tool

        result = run(call_tool("pdf_read", {"path": "/nonexistent/file.pdf"}))
        data = json.loads(result[0].text)
        assert "error" in data

    def test_non_pdf_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")

        from servers.pdf_server import call_tool

        result = run(call_tool("pdf_read", {"path": str(f)}))
        data = json.loads(result[0].text)
        assert "error" in data


# ── web_scrape ────────────────────────────────────────────────────────────────

class TestWebScrape:
    def test_invalid_url(self):
        from servers.web_server import call_tool

        result = run(call_tool("web_scrape", {"url": "http://localhost:1/nonexistent"}))
        data = json.loads(result[0].text)
        assert "error" in data


# ── code_exec ─────────────────────────────────────────────────────────────────

class TestCodeExec:
    def test_simple_print(self):
        from servers.code_server import call_tool

        result = run(call_tool("code_exec", {"code": "print('hello world')"}))
        data = json.loads(result[0].text)
        assert data["exit_code"] == 0
        assert "hello world" in data["stdout"]

    def test_pandas_computation(self):
        code = """
import pandas as pd
df = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
print(df["a"].sum())
"""
        from servers.code_server import call_tool

        result = run(call_tool("code_exec", {"code": code}))
        data = json.loads(result[0].text)
        assert data["exit_code"] == 0
        assert "6" in data["stdout"]

    def test_timeout_enforcement(self):
        code = "import time; time.sleep(60)"
        from servers.code_server import call_tool

        result = run(call_tool("code_exec", {"code": code, "timeout_seconds": 2}))
        data = json.loads(result[0].text)
        assert data["exit_code"] == -1
        assert "timed out" in data["stderr"].lower()

    def test_syntax_error_captured(self):
        from servers.code_server import call_tool

        result = run(call_tool("code_exec", {"code": "def bad syntax here"}))
        data = json.loads(result[0].text)
        assert data["exit_code"] != 0

    def test_network_blocked(self):
        code = "import socket; socket.create_connection(('8.8.8.8', 53))"
        from servers.code_server import call_tool

        result = run(call_tool("code_exec", {"code": code}))
        data = json.loads(result[0].text)
        # Should fail — either import blocked or connection refused
        assert data["exit_code"] != 0 or "error" in data["stderr"].lower()


# ── vector_search ─────────────────────────────────────────────────────────────

class TestVectorSearch:
    def test_empty_collection_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.cfg.CHROMA_PERSIST_DIR", tmp_path)
        monkeypatch.setattr("config.cfg.CHROMA_COLLECTION", "test_col")

        from servers.vector_server import call_tool

        result = run(call_tool("vector_search", {"query": "test query"}))
        data = json.loads(result[0].text)
        assert "documents" in data
        assert data["documents"] == []


# ── intent classifier ─────────────────────────────────────────────────────────

class TestIntentClassifier:
    def test_url_triggers_web_scrape(self):
        from utils.intent_classifier import classify

        calls = classify("Scrape https://example.com and summarise")
        assert any(c.tool == "web_scrape" for c in calls)

    def test_pdf_path_triggers_pdf_read(self):
        from utils.intent_classifier import classify

        calls = classify("Read /home/user/report.pdf and summarise")
        assert any(c.tool == "pdf_read" for c in calls)

    def test_csv_path_triggers_file_search(self):
        from utils.intent_classifier import classify

        calls = classify("Analyse ~/data/sales.csv")
        assert any(c.tool == "file_search" for c in calls)

    def test_memory_keyword_triggers_vector_search(self):
        from utils.intent_classifier import classify

        calls = classify("Do you remember what we discussed earlier?")
        assert any(c.tool == "vector_search" for c in calls)

    def test_plain_query_returns_empty(self):
        from utils.intent_classifier import classify

        calls = classify("What is the capital of France?")
        assert calls == []


# ── short-term memory ─────────────────────────────────────────────────────────

class TestShortTermMemory:
    def test_add_and_retrieve(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.cfg.SESSION_MEMORY_DIR", tmp_path)
        from memory.short_term import ShortTermMemory

        mem = ShortTermMemory(session_id="test")
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        msgs = mem.messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    def test_maxlen_respected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.cfg.SESSION_MEMORY_DIR", tmp_path)
        monkeypatch.setattr("config.cfg.SHORT_TERM_MEMORY_SIZE", 3)
        from memory.short_term import ShortTermMemory

        mem = ShortTermMemory(session_id="test2")
        for i in range(10):
            mem.add("user", f"Message {i}")
        assert len(mem) == 3

    def test_persist_and_reload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.cfg.SESSION_MEMORY_DIR", tmp_path)
        from memory.short_term import ShortTermMemory

        mem = ShortTermMemory(session_id="reload_test")
        mem.add("user", "Persistent message")
        mem.save()

        mem2 = ShortTermMemory(session_id="reload_test")
        assert any(m["content"] == "Persistent message" for m in mem2.messages())
