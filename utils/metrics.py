"""
utils/metrics.py

In-process metrics store for the API layer.
Tracks request counts, tool-call counts, latency distribution, and errors.
Exposed via GET /metrics without any external dependency.

All operations are thread-safe (asyncio single-thread model means no
explicit locking is needed, but the use of plain lists is intentional for
simplicity — swap for a deque with maxlen if you want bounded memory).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)

_METRICS_FILE = Path("./data/metrics.json")


class MetricsStore:
    """
    Lightweight in-process metrics collector.
    Persists a snapshot to data/metrics.json on each update so metrics
    survive restarts and are inspectable without hitting the API.
    """

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.total_tool_calls: int = 0
        self.errors: int = 0
        self._latencies: list[float] = []
        self._start_time: float = time.time()
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if _METRICS_FILE.exists():
            try:
                data = json.loads(_METRICS_FILE.read_text())
                self.total_requests = data.get("total_requests", 0)
                self.total_tool_calls = data.get("total_tool_calls", 0)
                self.errors = data.get("errors", 0)
                # Latencies are not persisted (they're in-memory only)
            except Exception:
                pass

    def _save(self) -> None:
        try:
            _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _METRICS_FILE.write_text(
                json.dumps(
                    {
                        "total_requests": self.total_requests,
                        "total_tool_calls": self.total_tool_calls,
                        "errors": self.errors,
                        "avg_latency_seconds": self.avg_latency,
                        "p95_latency_seconds": self.p95_latency,
                        "uptime_seconds": round(time.time() - self._start_time, 1),
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            log.warning(f"Could not persist metrics: {exc}")

    # ── Recording ─────────────────────────────────────────────────────────

    def record_request(self, latency: float) -> None:
        self.total_requests += 1
        self._latencies.append(latency)
        # Keep only the last 1000 latencies to bound memory
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]
        self._save()

    def record_tool_call(self) -> None:
        self.total_tool_calls += 1

    def record_error(self) -> None:
        self.errors += 1
        self._save()

    # ── Derived stats ─────────────────────────────────────────────────────

    @property
    def avg_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return round(sum(self._latencies) / len(self._latencies), 3)

    @property
    def p95_latency(self) -> float:
        if not self._latencies:
            return 0.0
        sorted_lats = sorted(self._latencies)
        idx = max(0, int(len(sorted_lats) * 0.95) - 1)
        return round(sorted_lats[idx], 3)

    def reset(self) -> None:
        """Reset all counters — useful in tests."""
        self.total_requests = 0
        self.total_tool_calls = 0
        self.errors = 0
        self._latencies = []


# Singleton — imported by orchestrator.py and api.py
metrics_store = MetricsStore()
