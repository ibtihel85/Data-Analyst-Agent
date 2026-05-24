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

def _is_schema_node(value: Any) -> bool:
    """
    Return True if *value* looks like a JSON Schema property descriptor
    (i.e. the LLM echoed the schema back instead of filling in a real value).
    A schema node is a dict that contains 'type' and/or 'description' keys
    but NOT any data-like keys.
    """
    if not isinstance(value, dict):
        return False
    schema_keys = {"type", "description", "default", "enum", "items", "properties"}
    non_schema_keys = set(value.keys()) - schema_keys
    has_schema_marker = "type" in value or "description" in value
    return has_schema_marker and not non_schema_keys


def _arguments_are_schema_leaked(arguments: dict) -> bool:
    """Return True if any argument value is a schema node (schema leakage)."""
    return any(_is_schema_node(v) for v in arguments.values())


def _extract_json_objects(text: str) -> list[dict]:
    """
    Extract all top-level JSON objects from *text* using a balanced-brace
    scanner.  Handles nested objects and arrays that defeat simple regexes.
    """
    results: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        fragment = text[i : j + 1]
                        try:
                            obj = json.loads(fragment)
                            if isinstance(obj, dict):
                                results.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return results


def _extract_tool_calls_from_text(text: str) -> list[ToolCall]:
    """
    Fallback parser for models that embed tool calls as JSON inside text
    rather than using Ollama's native tool_calls field.

    Handles fenced ```json blocks and bare JSON objects anywhere in the text.
    Uses a balanced-brace scanner so nested argument objects are parsed
    correctly (the previous regex approach silently dropped nested values).
    """
    calls: list[ToolCall] = []

    # Prefer fenced blocks; fall back to scanning the whole text
    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
    search_texts = fenced_blocks if fenced_blocks else [text]

    known_tool_names = {
        "file_search", "pdf_read", "web_scrape", "vector_search", "code_exec"
    }

    for source in search_texts:
        for obj in _extract_json_objects(source):
            name = obj.get("name") or obj.get("tool") or obj.get("function", {}).get("name")
            if not isinstance(name, str) or name not in known_tool_names:
                continue
            args = (
                obj.get("arguments")
                or obj.get("parameters")
                or obj.get("function", {}).get("arguments")
                or {}
            )
            if not isinstance(args, dict):
                continue
            if _arguments_are_schema_leaked(args):
                log.warning(
                    f"[ollama] Fallback parser: schema leakage detected in '{name}' "
                    f"— discarding malformed tool call"
                )
                continue
            calls.append(ToolCall(id=f"fallback_{len(calls)}", name=name, arguments=args))

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
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}

            # Detect schema leakage: model echoed property descriptors as values
            if _arguments_are_schema_leaked(args):
                log.warning(
                    f"[ollama] Schema leakage detected in native tool call '{name}' "
                    f"— dropping malformed call so retry logic can recover"
                )
                continue          # skip this tool call entirely; do NOT append it

            tool_calls.append(
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=name,
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
