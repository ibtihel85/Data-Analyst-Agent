"""
tests/test_additions.py

Unit tests for the four additions:
  - MetricsStore (utils/metrics.py)
  - Ingest pipeline (ingest.py)
  - API schemas (api.py — import-only check)
  - Eval case definitions (tests/eval_suite.py — import-only check)

These tests do NOT require Ollama, a live API server, or ChromaDB writes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── MetricsStore ──────────────────────────────────────────────────────────────

class TestMetricsStore:
    def test_initial_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        assert m.total_requests == 0
        assert m.total_tool_calls == 0
        assert m.errors == 0

    def test_record_request_increments_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        m.record_request(latency=1.5)
        m.record_request(latency=2.5)
        assert m.total_requests == 2

    def test_avg_latency(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        m.record_request(latency=1.0)
        m.record_request(latency=3.0)
        assert m.avg_latency == 2.0

    def test_p95_latency_single(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        m.record_request(latency=5.0)
        assert m.p95_latency == 5.0

    def test_tool_call_counter(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        m.record_tool_call()
        m.record_tool_call()
        m.record_tool_call()
        assert m.total_tool_calls == 3

    def test_error_counter(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        m.record_error()
        assert m.errors == 1

    def test_persist_and_reload(self, tmp_path, monkeypatch):
        metrics_path = tmp_path / "metrics.json"
        monkeypatch.setattr("utils.metrics._METRICS_FILE", metrics_path)
        from utils.metrics import MetricsStore
        m = MetricsStore()
        m.record_request(latency=2.0)
        m.record_tool_call()
        # Reload
        m2 = MetricsStore()
        assert m2.total_requests == 1
        assert m2.total_tool_calls == 1

    def test_reset(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        m.record_request(latency=1.0)
        m.reset()
        assert m.total_requests == 0
        assert m.avg_latency == 0.0

    def test_latency_buffer_bounded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.metrics._METRICS_FILE", tmp_path / "metrics.json")
        from utils.metrics import MetricsStore
        m = MetricsStore()
        for i in range(1200):
            m._latencies.append(float(i))
        m.record_request(latency=9999.0)
        assert len(m._latencies) <= 1001  # 1000 kept + 1 new before trim


# ── Ingest text extraction ────────────────────────────────────────────────────

class TestIngestExtraction:
    def test_read_txt_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Hello world\nSecond line", encoding="utf-8")
        from ingest import extract_text
        text = extract_text(f)
        assert "Hello world" in text
        assert "Second line" in text

    def test_read_md_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\nSome content here.", encoding="utf-8")
        from ingest import extract_text
        text = extract_text(f)
        assert "Title" in text

    def test_read_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,score\nAlice,90\nBob,85", encoding="utf-8")
        from ingest import extract_text
        text = extract_text(f)
        assert "name" in text.lower() or "Alice" in text

    def test_discover_files_single(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        from ingest import discover_files
        found = discover_files(f)
        assert len(found) == 1
        assert found[0] == f

    def test_discover_files_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "c.exe").write_text("ignored")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.csv").write_text("x,y\n1,2")
        from ingest import discover_files
        found = discover_files(tmp_path)
        names = {f.name for f in found}
        assert "a.txt" in names
        assert "b.md" in names
        assert "d.csv" in names
        assert "c.exe" not in names

    def test_discover_unsupported_file(self, tmp_path):
        f = tmp_path / "binary.exe"
        f.write_bytes(b"\x00\x01\x02")
        from ingest import discover_files
        found = discover_files(f)
        assert found == []

    def test_discover_nonexistent(self, tmp_path):
        from ingest import discover_files
        found = discover_files(tmp_path / "does_not_exist.txt")
        assert found == []


# ── Eval suite imports + scoring ──────────────────────────────────────────────

class TestEvalSuite:
    def test_eval_cases_defined(self):
        from tests.eval_suite import EVAL_CASES
        assert len(EVAL_CASES) >= 10

    def test_eval_case_has_required_fields(self):
        from tests.eval_suite import EVAL_CASES
        for case in EVAL_CASES:
            assert case.id
            assert case.query
            assert isinstance(case.expected_tools, list)
            assert isinstance(case.expected_keywords, list)

    def test_score_tool_hit(self):
        from tests.eval_suite import EvalResult, EvalCase, score_tool
        case = EvalCase("T1", "q", ["file_search"], ["file"])
        result = EvalResult("T1", "q", "resp", ["file_search"], 0, 0, 0, False)
        assert score_tool(result, case) == 1.0

    def test_score_tool_miss(self):
        from tests.eval_suite import EvalResult, EvalCase, score_tool
        case = EvalCase("T2", "q", ["file_search"], ["file"])
        result = EvalResult("T2", "q", "resp", ["code_exec"], 0, 0, 0, False)
        assert score_tool(result, case) == 0.0

    def test_score_keywords_full(self):
        from tests.eval_suite import EvalResult, EvalCase, score_keywords
        case = EvalCase("T3", "q", [], ["hello", "world"])
        result = EvalResult("T3", "q", "hello world response", [], 0, 0, 0, False)
        assert score_keywords(result, case) == 1.0

    def test_score_keywords_partial(self):
        from tests.eval_suite import EvalResult, EvalCase, score_keywords
        case = EvalCase("T4", "q", [], ["hello", "world", "missing"])
        result = EvalResult("T4", "q", "hello world response", [], 0, 0, 0, False)
        assert score_keywords(result, case) == pytest.approx(2 / 3, rel=0.01)

    def test_combined_score(self):
        from tests.eval_suite import combined_score
        assert combined_score(1.0, 1.0) == 1.0
        assert combined_score(0.0, 0.0) == 0.0
        # tool=1, kw=0.5 → 0.6 + 0.2 = 0.8
        assert combined_score(1.0, 0.5) == pytest.approx(0.8, rel=0.01)


# ── API import sanity check ───────────────────────────────────────────────────

class TestAPIImport:
    def test_api_schemas_importable(self):
        """api.py should be importable without a running Ollama / server."""
        # We only check that Pydantic models and the FastAPI app exist —
        # we don't start the server.
        try:
            from api import ChatRequest, ChatResponse, HealthResponse, MetricsResponse, app
            assert app is not None
        except Exception as exc:
            pytest.fail(f"api.py import failed: {exc}")

    def test_chat_request_validation(self):
        from api import ChatRequest
        req = ChatRequest(query="Hello agent")
        assert req.query == "Hello agent"
        assert req.session_id is None

    def test_chat_request_min_length(self):
        from pydantic import ValidationError
        from api import ChatRequest
        with pytest.raises(ValidationError):
            ChatRequest(query="")
