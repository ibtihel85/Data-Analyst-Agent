"""
memory/short_term.py

Short-term conversation memory backed by a Python deque.
Persists to a JSON file per session so context survives restarts.
"""
from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class ShortTermMemory:
    """
    Stores the last N conversation messages in a deque and persists
    them to a per-session JSON file.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self._buffer: deque[dict[str, Any]] = deque(maxlen=cfg.SHORT_TERM_MEMORY_SIZE)
        self._file: Path = cfg.SESSION_MEMORY_DIR / f"session_{self.session_id}.json"
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                for msg in data.get("messages", []):
                    self._buffer.append(msg)
                log.debug(f"Loaded {len(self._buffer)} messages from {self._file.name}")
            except Exception as exc:
                log.warning(f"Could not load session file: {exc}")

    def save(self) -> None:
        cfg.SESSION_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "saved_at": datetime.utcnow().isoformat(),
            "messages": list(self._buffer),
        }
        self._file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # ── Message management ────────────────────────────────────────────────

    def add(self, role: str, content: str, **extra: Any) -> None:
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(extra)
        self._buffer.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._buffer.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def add_assistant_tool_call(self, content: str, tool_calls_raw: list[dict]) -> None:
        """Add an assistant message that contains tool_calls."""
        self._buffer.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls_raw}
        )

    def messages(self) -> list[dict]:
        """Return the current buffer as a plain list (for Ollama)."""
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        if self._file.exists():
            self._file.unlink()

    def __len__(self) -> int:
        return len(self._buffer)
