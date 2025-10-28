# Question Answering Microservice

A production-ready FastAPI microservice that answers natural language questions using Retrieval-Augmented Generation (RAG). The service indexes documents into a vector database (ChromaDB) and combines semantic search with OpenAI's language models to provide accurate, source-cited answers.

## Features

- **Semantic Search**: ChromaDB-powered vector similarity search for relevant context retrieval
- **Source Attribution**: Every answer cites specific document chunks for transparency
- **Streaming Support**: Optional streaming responses for real-time answer generation
- **Robust Error Handling**: Unified error format with automatic retry/backoff for transient failures
- **Production-Ready**: Comprehensive logging, validation, and exception handling

## Quick Start

```bash
# Install dependencies (requires uv: https://github.com/astral-sh/uv)
uv sync

# Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Build the document index
make index

# Start the API server
make dev
```

The API will be available at `http://localhost:8000`.

## Documentation

- **[Getting Started](docs/getting-started.md)** - Setup, configuration, and first steps
- **[API Reference](docs/api.md)** - Endpoints, request/response schemas, error handling
- **[Examples](docs/examples.md)** - Sample requests and responses
- **[Architecture](docs/architecture.md)** - System design, data flow, and components
- **[Directory Structure](docs/directory-structure.md)** - Project layout and file purposes

## Technology Stack

- **FastAPI** - Modern async web framework
- **ChromaDB** - Vector database for embeddings
- **OpenAI API** - Embeddings and text generation
- **Tenacity** - Retry/backoff for resilience
- **Pydantic** - Request/response validation