# 🔬 Local Data Analyst Agent

A **100% local, CPU-only, open-source** data analyst agent powered by:
- **Ollama** for local LLM inference (no cloud, no API keys)
- **MCP (Model Context Protocol)** for structured tool calling
- **ChromaDB + sentence-transformers** for semantic memory
- **pandas / matplotlib / DuckDB** for data analysis

---

## Architecture

```
User Query
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
    │         ┌─────┴─────────────────────────────────┐
    │         │ filesystem_server │ pdf_server          │
    │         │ web_server        │ vector_server        │
    │         │ code_server       │                      │
    │         └───────────────────────────────────────┘
    │               │
    └── tool results injected → LLM continues
    │
    ▼
[Final Response]
    │
    ▼
[Memory: short-term deque + ChromaDB long-term]
```

### Project Structure

```
local-data-analyst/
├── orchestrator.py          ← main entry point / agent loop
├── config.py                ← all settings (reads .env)
├── requirements.txt
├── .env.example
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
│   └── logger.py            ← Rich-based logger
│
├── data/
│   ├── chroma_store/        ← ChromaDB persisted here (auto-created)
│   └── sessions/            ← Short-term memory JSON (auto-created)
│
└── tests/
    ├── test_tools.py
    └── test_ollama_client.py
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

## Setup (CPU-only)

### 1. Install Ollama

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**  
Download from https://ollama.com/download

### 2. Pull a CPU-safe model

The agent is configured for `llama3.2:3b` by default (fastest on CPU, ~2 GB).

```bash
# Recommended: smallest, fastest on CPU
ollama pull llama3.2:3b

# Alternative: better reasoning, slower (~4 GB)
ollama pull mistral:7b-instruct-q4_0

# Alternative: Microsoft Phi-3 mini
ollama pull phi3:mini
```

Start Ollama (if not already running as a service):
```bash
ollama serve
```

### 3. Clone / open the project

```bash
cd local-data-analyst
```

### 4. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# local-data-analyst\Scripts\activate         # Windows PowerShell
```

### 5. Install CPU-only PyTorch + dependencies

> **Important:** The `--extra-index-url` flag ensures you get the CPU-only build of PyTorch (~200 MB instead of ~3 GB with CUDA).

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 6. Configure environment

```bash
cp .env.example .env
# Edit .env if you want a different model or paths
```

### 7. Download the embedding model (one-time, ~80 MB)

The `all-MiniLM-L6-v2` model downloads automatically on first use. To pre-download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## Running the Agent

```bash
python orchestrator.py
```

The agent will:
1. Check Ollama is running and the model is available
2. Start MCP server subprocesses on demand
3. Open an interactive CLI prompt

---

## Example Queries

### Analyse a local CSV file
```
You: I have a file at ~/data/sales_2024.csv — analyse the monthly revenue trend
```

### Read a PDF
```
You: Read C:\Users\ibtih\Downloads\q3_report.pdf and summarise the key metrics
```

### Scrape a web page
```
You: Scrape https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal) and show me the top 10 countries
```

### Find files
```
You: Find all CSV files in my C:\Users\ directory
```

### Run a computation
```
You: Generate 1000 random numbers, compute their mean and standard deviation, and plot a histogram saved to /tmp/hist.png
```

### Memory retrieval
```
You: Do you remember what we analysed earlier?
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- All 5 tool implementations
- Intent classifier
- Short-term memory (deque + persistence)
- Ollama tool-call parser (mocked)

> Tests do **not** require Ollama or a live network connection.

---

## Changing the Model

Edit `.env`:

```
OLLAMA_MODEL=mistral:7b-instruct-q4_0
```

Or set it inline:
```bash
OLLAMA_MODEL=phi3:mini python orchestrator.py
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

## Offline Operation

After initial setup (pip install + `ollama pull`), the agent runs **fully offline**:

| Component | Offline? |
|-----------|---------|
| LLM (Ollama) | ✅ Local inference |
| Embeddings (MiniLM) | ✅ Cached in `~/.cache/torch/` |
| ChromaDB | ✅ Local file store |
| web_scrape tool | ⚠️ Requires internet (by design) |
| All other tools | ✅ Local only |

---

## Troubleshooting

**"Ollama is not reachable"**  
→ Run `ollama serve` in a separate terminal.

**"Model not found"**  
→ Run `ollama pull llama3.2:3b`

**Slow inference on CPU**  
→ Use `llama3.2:3b` (fastest). Expect 1–5 tokens/sec on a 4-core CPU.  
→ Reduce `num_ctx` in `llm/ollama_client.py` if RAM is tight.

**ChromaDB import errors**  
→ Ensure `torch` CPU-only was installed before `chromadb`.

**Tool timeout errors**  
→ Increase `TOOL_TIMEOUT_SECONDS` in `.env`.
