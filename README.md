# 🔬 Local Data Analyst Agent

A **100% local, CPU-only, open-source** data analyst agent powered by:
- **Ollama** for local LLM inference (no cloud, no API keys)
- **MCP (Model Context Protocol)** for structured tool calling
- **ChromaDB + sentence-transformers** for semantic memory and RAG
- **FastAPI** REST API so the agent is usable as a service
- **pandas / matplotlib / DuckDB** for data analysis

---

## Architecture

```
User Query  (CLI  or  POST /chat)
    │
    ▼
[Intent Classifier]         ← regex/keyword pre-routing
    │
    ▼
[Orchestrator Loop]
    │
    ├── LLM turn (Ollama /api/chat + tools array)
    │       │
    │       └── tool_calls detected
    │               │
    │               ▼
    │         [MCP Dispatcher]  ← JSON-RPC over stdio subprocess
    │               │
    │         ┌─────┴──────────────────────────────────────┐
    │         │ filesystem_server │ pdf_server              │
    │         │ web_server        │ vector_server (RAG)     │
    │         │ code_server       │                         │
    │         └────────────────────────────────────────────┘
    │               │
    └── tool results injected → LLM continues
    │
    ▼
[Final Response]
    │
    ├── Short-term memory  (sliding deque, JSON-persisted)
    └── Long-term memory   (ChromaDB vector store, cosine similarity)

                ↑
         ingest.py pre-indexes any local file into the vector store
```

### Project Structure

```
local-data-analyst/
├── orchestrator.py          ← main CLI entry point / agent loop
├── api.py                   ← FastAPI REST API (NEW)
├── ingest.py                ← document ingestion CLI for RAG (NEW)
├── config.py                ← all settings (reads .env)
├── requirements.txt
├── .env.example
├── Dockerfile               ← production container (NEW)
├── docker-compose.yml       ← Ollama + agent in one command (NEW)
│
├── llm/
│   └── ollama_client.py     ← async Ollama wrapper + tool-call parser
│
├── mcp_client/
│   └── dispatcher.py        ← JSON-RPC stdio dispatcher
│
├── servers/                 ← MCP tool servers (stdio subprocesses)
│   ├── filesystem_server.py ← tool: file_search
│   ├── pdf_server.py        ← tool: pdf_read
│   ├── web_server.py        ← tool: web_scrape
│   ├── vector_server.py     ← tool: vector_search
│   └── code_server.py       ← tool: code_exec
│
├── memory/
│   ├── short_term.py        ← deque + JSON persistence
│   └── long_term.py         ← ChromaDB + MiniLM embeddings
│
├── tools/
│   └── schemas.py           ← JSON schemas sent to Ollama tools array
│
├── utils/
│   ├── intent_classifier.py ← regex-based pre-routing
│   ├── logger.py            ← Rich-based logger
│   └── metrics.py           ← in-process latency + usage metrics (NEW)
│
├── data/
│   ├── chroma_store/        ← ChromaDB persisted here (auto-created)
│   ├── sessions/            ← Short-term memory JSON (auto-created)
│   └── metrics.json         ← persisted metrics snapshot (auto-created)
│
└── tests/
    ├── test_tools.py
    ├── test_ollama_client.py
    ├── test_additions.py    ← tests for metrics, ingest, eval (NEW)
    └── eval_suite.py        ← integration eval pipeline (NEW)
```

---

## Requirements

| Component | Minimum |
|-----------|---------|
| RAM | 8 GB (16 GB recommended) |
| CPU | 4-core (8-core recommended) |
| Disk | ~5 GB (models + libraries) |
| GPU | **Not required** |
| Python | 3.11+ |

---

## Option A — Run locally (CLI or API)

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve
```

### 2. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env to change model or paths if needed
```

### 4a. Run the interactive CLI

```bash
python orchestrator.py
```

### 4b. Run the REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

---

## Option B — Docker (Ollama + agent in one command)

```bash
# Build and start everything
docker compose up --build

# Pull the model (first time only)
docker compose exec ollama ollama pull llama3.2:3b

# The agent API is now available at http://localhost:8000
```

Other Docker commands:

```bash
# Interactive CLI inside container
docker compose run --rm agent python orchestrator.py

# Ingest documents
docker compose run --rm agent python ingest.py /app/data/host_data

# Run eval suite
docker compose run --rm agent python tests/eval_suite.py
```

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Run a query through the full agent loop |
| `GET` | `/health` | Liveness + Ollama status |
| `GET` | `/memory/search?q=...` | Semantic search over long-term memory |
| `GET` | `/metrics` | Latency + usage statistics |
| `DELETE` | `/memory` | Clear all long-term memory |

### Example: POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Compute the mean of [10, 20, 30] in Python"}'
```

```json
{
  "response": "The mean of [10, 20, 30] is 20.0.",
  "session_id": "a1b2c3d4",
  "duration_seconds": 4.231,
  "tool_calls_made": 1
}
```

---

## RAG: Ingesting Documents

Pre-index local documents into ChromaDB so the agent can retrieve them
during analysis:

```bash
# Index a single file
python ingest.py ~/reports/q3_report.pdf

# Index a whole folder (txt, md, csv, pdf)
python ingest.py ~/data/

# Preview without writing
python ingest.py ~/data/ --dry-run

# List what is stored
python ingest.py --list

# Clear the store
python ingest.py --clear
```

After ingestion, ask the agent:

```
You: Summarise the Q3 report findings
```

The agent will call `vector_search` to retrieve relevant chunks before answering.

---

## Running Tests

```bash
# Unit tests (no Ollama required)
pytest tests/test_tools.py tests/test_ollama_client.py tests/test_additions.py -v

# Integration eval (requires Ollama running)
python tests/eval_suite.py

# Eval with a specific model
python tests/eval_suite.py --model mistral:7b-instruct-q4_0

# Save eval results to JSON
python tests/eval_suite.py --output eval_results.json

# Run only specific eval cases
python tests/eval_suite.py --cases E01 E02 E05
```

The eval suite scores each of 10 fixed queries on:
- **Tool accuracy** (60%) — were the expected tools called?
- **Keyword coverage** (40%) — does the response contain expected terms?

---

## Metrics

The agent tracks latency and usage automatically. Access via:

```bash
# API
curl http://localhost:8000/metrics

# Direct file (also written to disk on every request)
cat data/metrics.json
```

---

## How MCP Works Here

Each tool runs as a **Python subprocess** communicating via **stdio JSON-RPC**:

1. `orchestrator.py` creates an `MCPDispatcher`
2. When a tool is needed, the dispatcher spawns the relevant `servers/*.py` subprocess (if not already running)
3. A JSON-RPC `tools/call` request is sent via stdin
4. The server executes the tool and writes the result to stdout
5. The dispatcher reads the response and injects it into the conversation

Servers are kept alive across multiple tool calls within a session to avoid subprocess startup overhead.

---

## Changing the Model

```bash
# In .env
OLLAMA_MODEL=mistral:7b-instruct-q4_0

# Or inline
OLLAMA_MODEL=phi3:mini python orchestrator.py
OLLAMA_MODEL=phi3:mini uvicorn api:app --port 8000
```

---

## Offline Operation

After initial setup, the agent runs **fully offline**:

| Component | Offline? |
|-----------|---------|
| LLM (Ollama) | ✅ Local inference |
| Embeddings (MiniLM) | ✅ Cached in `~/.cache/torch/` |
| ChromaDB | ✅ Local file store |
| web_scrape tool | ⚠️ Requires internet (by design) |
| All other tools | ✅ Local only |
