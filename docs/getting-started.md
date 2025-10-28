# Getting Started

This guide walks you through setting up and running the Question Answering Microservice.

## Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package installer (recommended)
- **OpenAI API Key** - Get one at [platform.openai.com](https://platform.openai.com)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wolf2627/question-answering-microservice.git
cd question-answering-microservice
```

### 2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies

```bash
uv sync
```

This creates a virtual environment and installs all required packages defined in `pyproject.toml`.

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional - Document Processing
DOCS_PATH=./documents              # Path to documents to index
CHUNK_SIZE=50                      # Words per chunk
CHUNK_OVERLAP=5                   # Overlapping words between chunks
EMBED_BATCH_SIZE=30                # Batch size for embedding generation

# Optional - Vector Store
CHROMA_PATH=./embeddings           # ChromaDB persistence directory
COLLECTION_NAME=documents          # Collection name in ChromaDB
EMBED_MODEL=text-embedding-3-large # OpenAI embedding model

# Optional - API Behavior
TOP_K=10                           # Number of chunks to retrieve
RESPONSE_MODEL=gpt-4.1-nano        # OpenAI model for answer generation
RESPONSE_MAX_TOKENS=300            # Max tokens in generated answer
RESPONSE_TEMPERATURE=0.0           # Generation temperature (0 = deterministic)

# Optional - Reliability
OPENAI_TIMEOUT=30.0                # Request timeout in seconds
MAX_EMBED_RETRIES=5                # Retry attempts for failed requests
STREAM_IDLE_TIMEOUT=60.0           # Streaming idle timeout
STREAM_MAX_DURATION=600.0          # Max streaming duration
```

### Supported Document Types

Place your documents in the `documents/` directory (or path specified by `DOCS_PATH`):

- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents
- `.pptx` - PowerPoint presentations

## Building the Document Index

Before starting the API, index your documents:

```bash
make index
```

Or directly:

```bash
uv run python -m src.ingest
```

This:
1. Loads all supported documents from `documents/`
2. Splits them into chunks (respecting `CHUNK_SIZE` and `CHUNK_OVERLAP`)
3. Generates embeddings using OpenAI's embedding model
4. Stores embeddings in ChromaDB at `embeddings/`

**Progress Output:**
```
Processing file: documents/example.pdf
Built 45 chunks from 1 documents
Embedded 30 items in current batch
Indexed 45/45 chunks (100%)
Completed ingestion for 1 documents
```

## Running the API

### Development Mode (with auto-reload)

```bash
make dev
```

Or:

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Base URL:** `http://localhost:8000`

### Production Mode

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Compose (Containerized)

The repository ships with a production-ready `Dockerfile`, `.dockerignore`, and `docker-compose.yml`.

1. Make sure your `.env` file is present (it is mounted into the container).
2. Build and start the stack:

  ```bash
  docker compose up --build
  ```

  The container entrypoint runs `make index` before starting the API, ensuring your documents are embedded on boot. The service becomes available at `http://localhost:8000`.

3. To run in detached mode:

  ```bash
  docker compose up --build -d
  ```

4. To stop and clean up containers:

  ```bash
  docker compose down
  ```

#### Bind Mounts

- `./documents` ➜ `/app/documents` (read-only): update your local docs without rebuilding the image.
- `./embeddings` ➜ `/app/embeddings`: preserves the ChromaDB index between restarts.

If you do not need persistence, remove or adjust the volume mappings in `docker-compose.yml`.

## Verifying the Setup

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok"
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?"}'
```

**Response:**
```json
{
  "answer": "The main topic is... (citing example.pdf__chunk_0)",
  "sources": [
    {
      "chunk_id": "example.pdf__chunk_0",
      "document": "example.pdf",
      "score": 0.87
    }
  ]
}
```

## Common Issues

### Missing API Key

**Error:**
```
RuntimeError: OPENAI_API_KEY is not configured
```

**Solution:** Add `OPENAI_API_KEY` to your `.env` file.

### No Documents Found

**Error:**
```
Documents Directory ./documents does not exist.
```

**Solution:** Create the `documents/` directory and add at least one supported file.

### Empty Index

If queries return "No context passages were retrieved":
1. Ensure documents are in `documents/`
2. Re-run `make index`
3. Check `embeddings/` directory was created

## Next Steps

- See [API Reference](api.md) for detailed endpoint documentation
- Check [Examples](examples.md) for more query patterns
- Read [Architecture](architecture.md) to understand how the system works
