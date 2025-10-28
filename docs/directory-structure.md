# Directory Structure

Complete guide to the project's file organization, explaining the purpose of each directory and file.

## Project Root

```
question-answering/
├── .dockerignore             # Files/folders excluded from Docker build context
├── .env                      # Environment variables (not in git)
├── .env.example              # Template for environment configuration
├── .github/                  # GitHub-specific files
│   └── copilot-instructions.md  # AI coding agent guidance
├── .gitignore                # Git ignore patterns
├── .python-version           # Python version specification (for pyenv/uv)
├── .venv/                    # Virtual environment (created by uv)
├── Dockerfile                # Image definition (uses uv, runs entrypoint)
├── Makefile                  # Common development commands
├── README.md                 # Project overview and quick start
├── docs/                     # Detailed documentation
├── documents/                # Source documents to index
├── embeddings/               # ChromaDB vector database (generated)
├── main.py                   # Legacy entry point (not actively used)
├── docker-compose.yml        # Compose stack exposing API on :8000
├── docker-entrypoint.sh      # Container entrypoint (make index → make dev)
├── pyproject.toml            # Project metadata and dependencies
├── src/                      # Application source code
└── uv.lock                   # Locked dependency versions
```

---

## Configuration Files

### `.env`
**Purpose:** Environment-specific configuration (secrets, tunables)  
**Status:** Git-ignored (never commit)  
**Contains:**
- `OPENAI_API_KEY` (required)
- Optional overrides for models, paths, retry settings

**Example:**
```bash
OPENAI_API_KEY=sk-...
DOCS_PATH=./documents
CHUNK_SIZE=50
TOP_K=10
```

### `.env.example`
**Purpose:** Template showing all available environment variables  
**Status:** Committed to git  
**Usage:** Copy to `.env` and fill in actual values

### `pyproject.toml`
**Purpose:** Python project metadata, dependencies, build configuration  
**Format:** TOML (PEP 621 standard)  
**Key Sections:**
```toml
[project]
name = "assessment"
requires-python = ">=3.12"
dependencies = [
    "chromadb>=1.2.2",
    "fastapi>=0.120.1",
    "openai-agents>=0.4.2",
    # ...
]
```

**Why uv?** Modern, fast dependency resolver (100x faster than pip)

### `Makefile`
**Purpose:** Common developer commands  
**Targets:**
- `make dev` - Start API server with hot reload
- `make index` - Build document index

**Why Makefile?** Discoverable, self-documenting commands for common tasks

---

## Source Code (`src/`)

### `src/api.py` (335 lines)
**Purpose:** FastAPI application, HTTP endpoints, request handling  
**Key Components:**

- **Models:**
  - `AskRequest` - Request validation schema
  - `AskResponse` - Response format
  - `SourceAttribution` - Source citation structure

- **Endpoints:**
  - `POST /ask` - Main question answering endpoint
  - `GET /health` - Health check

- **Exception Handlers:**
  - `validation_exception_handler` - Pydantic validation errors (422)
  - `http_exception_handler` - Raised HTTPExceptions
  - `retry_exception_handler` - Tenacity retry exhaustion (502)
  - `generic_exception_handler` - Catch-all (500)

- **Helper Functions:**
  - `_retrieve_context()` - Semantic search for relevant chunks
  - `_generate_answer()` - LLM-based answer generation
  - `_error_payload()` / `_error_response()` - Unified error formatting
  - `_serialize_event()` - Streaming event formatting

**Dependencies:** FastAPI, Pydantic, Tenacity

### `src/config.py` (45 lines)
**Purpose:** Configuration management, environment variable loading  
**Key Components:**

- **`Settings` Dataclass:**
  - Immutable (`frozen=True`)
  - Type-safe fields with defaults
  - Loads from environment via `os.getenv()`

- **`get_settings()` Function:**
  - Returns cached singleton instance (`@lru_cache`)
  - Ensures settings loaded once per process

**Environment Variables:**
```python
docs_path: Path              # DOCS_PATH (default: ./documents)
chunk_size: int              # CHUNK_SIZE (default: 50)
openai_api_key: str          # OPENAI_API_KEY (required)
embed_model: str             # EMBED_MODEL (default: text-embedding-3-large)
top_k: int                   # TOP_K (default: 10)
# ... and more
```

### `src/vector_store.py` (140 lines)
**Purpose:** ChromaDB vector database abstraction  
**Key Components:**

- **`DocumentChunk` Dataclass:**
  - Represents a single chunk pre-indexing
  - Fields: `chunk_id`, `document_id`, `source_path`, `chunk_index`, `content`

- **`RetrievedChunk` Dataclass:**
  - Represents a chunk retrieved from vector search
  - Fields: `chunk_id`, `content`, `score`, `metadata`

- **`VectorStore` Class:**
  - `__init__()` - Creates/connects to persistent ChromaDB
  - `upsert()` - Adds/updates chunks with embeddings
  - `similarity_search()` - Queries by embedding vector
  - `similarity_search_text()` - Queries by text (embeds internally)

- **`_distance_to_similarity()` Function:**
  - Converts ChromaDB distance metrics to `[0,1]` similarity scores
  - Handles cosine distance (0-1) and L2 distance (0-∞)

**ChromaDB Details:**
- **Backend:** SQLite (embedded)
- **Persistence Path:** `embeddings/` (configurable via `CHROMA_PATH`)
- **Collection Name:** `documents` (configurable via `COLLECTION_NAME`)

### `src/openai_client.py` (145 lines)
**Purpose:** OpenAI API wrapper with retry/backoff  
**Key Components:**

- **`OpenAIClientConfig` Dataclass:**
  - Configuration for API client
  - Fields: `api_key`, `embed_model`, `timeout`, `max_retries`, backoff settings

- **`OpenAIClient` Class:**
  - `embed_text()` - Single text embedding
  - `embed_texts()` - Batch text embeddings
  - `generate_answer()` - Synchronous completion
  - `generate_answer_stream()` - Async streaming completion (future)
  - `_execute_with_retry()` - Retry wrapper for sync operations
  - `_stream_with_retry()` - Retry wrapper for async streams

**Retry Logic:**
- Exponential backoff: 1s → 2s → 4s → 8s → 16s (capped at 30s)
- Random jitter to prevent thundering herd
- Retries on: 429 (rate limit), 5xx (server errors), timeouts
- Fails fast on: 400 (bad request), 401 (auth), 404 (not found)

### `src/ingest.py` (220 lines)
**Purpose:** Document loading, chunking, embedding, indexing  
**Key Components:**

- **`LoadedDocument` Dataclass:**
  - Represents a loaded document
  - Fields: `path`, `content`

- **Functions:**
  - `load_documents()` - Recursively loads all supported files from `docs_path`
  - `_read_text()` - Extracts text from `.txt`, `.md`, `.pdf`, `.pptx`
  - `chunk_text()` - Splits text into word-based overlapping chunks
  - `build_chunks()` - Converts loaded documents into `DocumentChunk` objects
  - `ingest_documents()` - Main orchestration: load → chunk → embed → upsert
  - `_batched()` - Utility for batching chunks

**Supported File Types:**
```python
SUPPORTED_FILE_TYPES = {".txt", ".md", ".pdf", ".pptx"}
```

**Chunk ID Format:**
```python
f"{safe_document_id}__chunk_{index}"
# Example: docs__api.md__chunk_3
```

**CLI Entry Point:**
```bash
python -m src.ingest
```

---

## Data Directories

### `documents/`
**Purpose:** Source documents to index  
**Contents:** User-provided files (`.txt`, `.md`, `.pdf`, `.pptx`)  
**Default:** Empty (users add their own documents)  
**Gitignore:** Typically ignored (documents are deployment-specific)

**Example Structure:**
```
documents/
├── product_docs.pdf
├── api_reference.md
└── internal/
    └── guidelines.txt
```

### `embeddings/`
**Purpose:** ChromaDB vector database persistence  
**Contents:** SQLite database and vector index files  
**Generated By:** `make index` / `src.ingest`  
**Gitignore:** Yes (regenerated from source documents)

**Structure:**
```
embeddings/
├── chroma.sqlite3                  # Metadata database
└── <uuid>/                         # Collection data
    ├── data_level0.bin
    ├── header.bin
    ├── index_metadata.pickle
    └── length.bin
```

**Regeneration:** Run `make index` to rebuild from scratch

---

## Documentation (`docs/`)

### `docs/getting-started.md`
**Purpose:** Setup, installation, first-run instructions  
**Audience:** New developers/users  
**Contents:**
- Prerequisites
- Installation steps
- Configuration guide
- Running the API
- Troubleshooting

### `docs/api.md`
**Purpose:** API reference documentation  
**Audience:** API consumers, frontend developers  
**Contents:**
- Endpoint specifications
- Request/response schemas
- Error codes and formats
- Rate limiting
- Interactive docs links

### `docs/examples.md`
**Purpose:** Practical usage examples  
**Audience:** Developers integrating the API  
**Contents:**
- Sample curl commands
- Python/JavaScript code snippets
- Streaming examples
- Error handling patterns
- Testing tips

### `docs/architecture.md`
**Purpose:** System design and technical decisions  
**Audience:** Contributors, maintainers, architects  
**Contents:**
- Component overview
- Data flow diagrams
- Design rationale
- Scalability considerations
- Security recommendations

### `docs/directory-structure.md` (this file)
**Purpose:** Project layout reference  
**Audience:** New contributors  
**Contents:** File-by-file purpose explanation

---

## GitHub Specific (`.github/`)

### `.github/copilot-instructions.md`
**Purpose:** Instructions for AI coding assistants (GitHub Copilot, Cursor, etc.)  
**Contents:**
- Architecture overview
- Core services and their responsibilities
- Configuration patterns
- Developer workflows
- Project-specific conventions

**Usage:** AI tools read this file to understand project context and provide better suggestions

---

## Other Files

### `main.py`
**Purpose:** Legacy entry point (not actively used)  
**Status:** Placeholder  
**Contents:** Simple "Hello from assessment!" print  
**Note:** May be removed or repurposed in future

### `uv.lock`
**Purpose:** Locked dependency versions for reproducible installs  
**Generated By:** `uv sync`  
**Format:** TOML-like (uv-specific)  
**Gitignore:** Typically committed (ensures team uses same versions)

### `.python-version`
**Purpose:** Specifies Python version for version managers (pyenv, asdf, uv)  
**Contents:** `3.12` (or similar)  
**Usage:** `uv` automatically uses this version when creating venv

### `.gitignore`
**Purpose:** Tells git which files to ignore  
**Common Entries:**
```
.env
.venv/
__pycache__/
embeddings/
*.pyc
.DS_Store
```

### `.dockerignore`
**Purpose:** Excludes local-only files from Docker build context  
**Key Entries:**
```
.git
.venv/
documents/
embeddings/
docker-compose.yml
```
**Why:** Keeps images lean and prevents sensitive data (e.g., `.env`) from being copied.

### `Dockerfile`
**Purpose:** Defines the application container image  
**Highlights:**
- Based on `python:3.12-slim`
- Installs `uv`, syncs dependencies via `uv sync --frozen --no-dev`
- Copies the repo (respecting `.dockerignore`)
- Grants execute permission to the entrypoint and exposes port `8000`

### `docker-entrypoint.sh`
**Purpose:** Container startup script  
**Steps:**
1. Runs `make index` to (re)build embeddings on boot
2. Executes `make dev` (starts FastAPI with auto-reload)

### `docker-compose.yml`
**Purpose:** Local orchestration / production-like stack  
**Features:**
- Builds the Dockerfile image
- Mounts `documents/` (read-only) and `embeddings/` (persistent)
- Loads environment variables from `.env`
- Publishes the API on `http://localhost:8000`

---

## File Naming Conventions

1. **Python Modules:** `snake_case.py` (e.g., `vector_store.py`)
2. **Configuration:** Dot-prefixed (e.g., `.env`, `.gitignore`)
3. **Documentation:** `kebab-case.md` (e.g., `getting-started.md`)
4. **Data Directories:** Lowercase, no underscores (e.g., `documents/`, `embeddings/`)

---

## Adding New Files: Guidelines

### New Python Module
**Location:** `src/`  
**Naming:** `descriptive_name.py`  
**Requirements:**
- Docstring at top explaining purpose
- Type hints for all functions
- Logger instance: `logger = logging.getLogger(__name__)`

**Example:**
```python
"""Module for handling user authentication."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

def authenticate_user(api_key: str) -> Optional[str]:
    """Validates API key and returns user ID."""
    # ...
```

### New Configuration Option
**Steps:**
1. Add field to `Settings` dataclass in `src/config.py`
2. Document in `docs/getting-started.md` (Environment Variables section)
3. Add to `.env.example` with descriptive comment
4. Update `.github/copilot-instructions.md` if it affects architecture

### New Documentation
**Location:** `docs/`  
**Naming:** `descriptive-name.md`  
**Requirements:**
- Link from `README.md` if important
- Clear audience statement at top
- Code examples where applicable
- Keep under 500 lines (split if longer)

---

## Common Patterns

### Imports
```python
# Standard library first
import asyncio
import logging
from pathlib import Path

# Third-party second
from fastapi import FastAPI
from pydantic import BaseModel

# Local last
from src.config import Settings
from src.vector_store import VectorStore
```

### Module-Level Logger
```python
logger = logging.getLogger(__name__)
```

### Type Hints
```python
def process_chunks(chunks: list[DocumentChunk]) -> dict[str, float]:
    """All functions have type hints."""
    # ...
```

### Dependency Injection (FastAPI)
```python
@app.get("/")
def endpoint(settings: Settings = Depends(get_settings)):
    # Settings injected, not imported directly
    pass
```

---

## Future Structure (If Refactored)

If `src/api.py` is split into modules:

```
src/
├── api/
│   ├── __init__.py          # FastAPI app factory
│   ├── dependencies.py      # Cached dependencies
│   ├── errors.py            # Exception handlers
│   ├── routers.py           # Endpoint definitions
│   ├── schemas.py           # Pydantic models
│   └── services.py          # Business logic
├── config.py
├── ingest.py
├── openai_client.py
└── vector_store.py
```

See [API Refactoring](#) section in Architecture docs for migration plan.
