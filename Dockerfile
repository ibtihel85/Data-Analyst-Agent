# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install CPU-only PyTorch first (separate layer for caching)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.3.0+cpu \
      --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System deps for pdfplumber and lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

# Create data directories
RUN mkdir -p data/sessions data/chroma_store

# Pre-download the embedding model so the first query doesn't block
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Expose API port
EXPOSE 8000

# Default: run the FastAPI server
# To run the CLI instead: docker run ... python orchestrator.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
