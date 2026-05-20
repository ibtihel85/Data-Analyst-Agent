"""
llm/ollama_client.py

Thin async wrapper around Ollama's /api/chat endpoint.
Handles:
  - tool-call JSON parsing
  - stateless request/response
  - retries with tenacity
  - CPU-only model validation
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)

# Ollama models known to be CPU-safe (quantised ≤4-bit or small param count)
CPU_SAFE_MODELS = {
    "llama3.2:3b",
    "llama3.2:1b",
    "mistral:7b-instruct-q4_0",
    "mistral:7b-instruct-q4_K_M",
    "phi3:mini",
    "phi3:3.8b",
    "gemma2:2b",
    "qwen2:1.5b",
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


def _extract_tool_calls_from_text(text: str) -> list[ToolCall]:
    """
    Fallback parser for models that embed tool calls as JSON inside text
    rather than using Ollama's native tool_calls field.

    Looks for patterns like:
      {"name": "tool_name", "arguments": {...}}
    or fenced ```json blocks.
    """
    calls: list[ToolCall] = []
    # Try fenced JSON blocks first
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = fenced or [text]

    for candidate in candidates:
        candidate = candidate.strip()
        # Find all JSON objects that look like tool calls
        for match in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}', candidate, re.DOTALL):
            try:
                obj = json.loads(match.group(0))
                if "name" in obj:
                    tc = ToolCall(
                        id=f"fallback_{len(calls)}",
                        name=obj["name"],
                        arguments=obj.get("arguments", obj.get("parameters", {})),
                    )
                    calls.append(tc)
            except json.JSONDecodeError:
                pass
    return calls


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = cfg.OLLAMA_BASE_URL
        self.model = cfg.OLLAMA_MODEL
        self._validate_model()

    def _validate_model(self) -> None:
        if self.model not in CPU_SAFE_MODELS:
            log.warning(
                f"Model '{self.model}' is not in the known CPU-safe list. "
                f"Proceeding anyway — ensure it is quantised (q4 or smaller)."
            )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Send a chat request to Ollama and return a parsed LLMResponse.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
                "num_ctx": 4096,  # conservative context for CPU
            },
        }
        if tools:
            payload["tools"] = tools

        log.debug(f"Sending {len(messages)} messages to Ollama ({self.model})")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        message: dict = data.get("message", {})
        content: str = message.get("content", "")
        raw_tool_calls: list[dict] = message.get("tool_calls", [])

        tool_calls: list[ToolCall] = []

        # Parse native Ollama tool_calls
        for i, tc in enumerate(raw_tool_calls):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        # Fallback: parse tool calls from text if none found natively
        if not tool_calls and content:
            tool_calls = _extract_tool_calls_from_text(content)

        finish = data.get("done_reason", "stop")

        log.debug(
            f"LLM response: {len(content)} chars, "
            f"{len(tool_calls)} tool calls, finish={finish}"
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish,
            raw=data,
        )

    async def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                r.raise_for_status()
                models = [m["name"] for m in r.json().get("models", [])]
                if self.model not in models:
                    log.warning(
                        f"Model '{self.model}' not found in Ollama. "
                        f"Available: {models}. Run: ollama pull {self.model}"
                    )
                    return False
                return True
        except Exception as exc:
            log.error(f"Ollama health check failed: {exc}")
            return False
