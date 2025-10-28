# Architecture

This document explains the system design, data flow, and key architectural decisions behind the Question Answering Microservice.

## Overview

The service implements **Retrieval-Augmented Generation (RAG)**, a pattern that combines:
1. **Semantic Search** - Finding relevant context from a vector database
2. **Language Model Generation** - Producing answers grounded in retrieved context

This architecture ensures answers are factual, traceable, and based on your specific documents rather than the model's training data.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Application(n8n)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Request Validation (Pydantic)                           │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Vector Store Service                                    │   │
│  │  • Embeds question via OpenAI                            │   │
│  │  • Queries ChromaDB for similar chunks                   │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Generation Service                                      │   │
│  │  • Calls OpenAI completion API                           │   │
│  │  • Formats context + question                            │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Response Formatting                                     │   │
│  │  • Builds answer + sources payload                       │   │ 
│  │  • Handles errors with unified format                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │                                    │
           │ Embeddings / responses             │ Vector Queries
           ▼                                    ▼
┌─────────────────────────┐      ┌─────────────────────────────┐
│    OpenAI API           │      │   ChromaDB (Persistent)     │
│  • text-embedding-3-*   │      │  • Cosine similarity        │
│  • gpt-4 / gpt-3.5      │      │  • SQLite backend           │
└─────────────────────────┘      └─────────────────────────────┘
```

## Core Components

### 1. API Layer (`src/api.py`)

**Responsibilities:**
- Exposes HTTP endpoints (`/ask`, `/health`)
- Validates incoming requests with Pydantic schemas
- Orchestrates retrieval → generation pipeline
- Handles errors with centralized exception handlers
- Supports both standard and streaming responses

**Key Design Decisions:**
- **FastAPI Dependency Injection:** Settings, clients, and stores are cached singletons injected via `Depends()`
- **Async/Await:** Heavy I/O (embedding, generation, vector search) runs in worker threads via `asyncio.to_thread()` to keep the event loop responsive
- **Unified Error Format:** All errors return `{"error": {"status", "message", "details"}}` for consistent client handling

### 2. Vector Store (`src/vector_store.py`)

**Responsibilities:**
- Wraps ChromaDB persistent client
- Converts query text → embeddings → similarity results
- Translates Chroma distances to `[0,1]` similarity scores

**Implementation Details:**
- **Distance Metric:** Cosine distance (0 = identical, 1 = opposite)
- **Similarity Conversion:** `similarity = 1 - distance` for cosine; `1 / (1 + distance)` for L2
- **Persistence:** SQLite-backed ChromaDB at `embeddings/` directory

**Why ChromaDB?**
- Lightweight, embedded vector database
- No separate server process required
- Ideal for prototypes and small-to-medium deployments
- Easy migration path to client-server mode for scaling

### 3. OpenAI Client (`src/openai_client.py`)

**Responsibilities:**
- Wraps OpenAI SDK with retry/backoff logic
- Provides unified interface for embeddings and completions
- Handles rate limits and transient failures

**Retry Strategy:**
- **Max Retries:** 5 attempts
- **Backoff:** Exponential (1s → 2s → 4s → 8s → 16s) with random jitter
- **Retryable Errors:** 429 (rate limit), 5xx (server errors), timeouts
- **Non-Retryable:** 400 (bad request), 401 (auth), 404 (not found)

**Why Custom Client?**
- OpenAI SDK retry logic is limited; custom implementation provides fine-grained control
- Centralized error handling for all API interactions
- Future extensibility (e.g., switching to Azure OpenAI, local models)

### 4. Configuration (`src/config.py`)

**Responsibilities:**
- Loads environment variables via `python-dotenv`
- Provides typed, immutable `Settings` dataclass
- Caches settings instance with `@lru_cache`

**Environment Variable Philosophy:**
- **Required:** `OPENAI_API_KEY` (fail fast if missing)
- **Sensible Defaults:** All other settings have production-ready defaults
- **12-Factor Compliance:** Configuration via environment, not code

### 5. Ingestion (`src/ingest.py`)

**Responsibilities:**
- Loads documents from `documents/` directory
- Chunks text into overlapping segments
- Generates embeddings in batches
- Upserts chunks into ChromaDB

**Chunking Strategy:**
- **Unit:** Words (not characters or tokens) for simplicity
- **Size:** 50 words per chunk (configurable via `CHUNK_SIZE`)
- **Overlap:** 5 words between chunks (configurable via `CHUNK_OVERLAP`)
- **Why Overlap?** Prevents important context from being split across chunk boundaries

**Chunk ID Format:**
```
{document_path_with_slashes_replaced_by_double_underscores}__chunk_{index}
```

Example: `docs__api.md__chunk_3`

**Why This Format?**
- Unique per chunk
- Embeds source document in ID
- Easy to parse for debugging
- Compatible with ChromaDB metadata

## Data Flow

### Indexing Flow (Offline)

```
documents/*.{txt,md,pdf,pptx}
    │
    ▼
Load Documents (src/ingest.py)
    │
    ▼
Chunk Text (50 words, 5 overlap)
    │
    ▼
Batch Chunks (30 per batch)
    │
    ▼
Generate Embeddings (OpenAI text-embedding-3-large)
    │
    ▼
Upsert to ChromaDB (embeddings/)
```

**Performance:**
- **Batch Size:** 30 chunks/batch to balance API limits and progress feedback
- **Parallelization:** Batches processed sequentially (future: parallel batches with rate limiting)
- **Progress Logging:** Prints percentage complete after each batch

### Query Flow (Online)

```
Client Request (question)
    │
    ▼
Validate Request (Pydantic)
    │
    ▼
Embed Question (OpenAI API)
    │
    ▼
Query ChromaDB (top_k=10)
    │
    ▼
Retrieve Chunks + Scores
    │
    ▼
Format Context Prompt:
  "Context passages:
   chunk_id_1 (source):
   {chunk_content_1}
   
   chunk_id_2 (source):
   {chunk_content_2}
   
   Question: {question}
   Respond with factual answer citing chunk IDs."
    │
    ▼
Generate Answer (OpenAI Completion)
    │
    ▼
Parse + Return:
  {
    "answer": "...",
    "sources": [...]
  }
```

**Latency Breakdown (typical):**
- Request validation: <1ms
- Question embedding: 200-500ms
- Vector search: 10-50ms
- Answer generation: 1-3s
- **Total:** ~2-4s per request

### Streaming Flow

```
Client Request (stream=true)
    │
    ▼
[Same retrieval as above]
    │
    ▼
Yield Event 1 (context):
  {"event": "context", "data": {"sources": [...]}}
    │
    ▼
Generate Answer (as before)
    │
    ▼
Yield Event 2 (answer):
  {"event": "answer", "data": {"answer": "...", "sources": [...]}}
```

**Current Limitation:** Generation is non-streaming (waits for full completion). Future enhancement: token-by-token streaming via OpenAI's streaming API.

## Error Handling & Resilience

### Exception Handler Hierarchy

```
1. RequestValidationError (422)
   ↓ Pydantic validation failures
   
2. StarletteHTTPException (varies)
   ↓ Explicitly raised HTTPExceptions
   
3. RetryError (502)
   ↓ Tenacity exhausted retries
   
4. Exception (500)
   ↓ Catch-all for unexpected errors
```

All handlers call `_error_response()` to ensure uniform payload structure.

### Retry Logic

Implemented via **Tenacity** library:

```python
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception(_should_retry),
)
```

**Retryable Conditions:**
- OpenAI rate limits (429)
- Server errors (5xx)
- Network timeouts

**Non-Retryable:**
- Client errors (4xx except 429)
- Validation errors
- Missing API keys

### Defensive Patterns

1. **Worker Threads:** Blocking I/O never runs on the event loop
2. **Exception Sanitization:** Internal exceptions never expose to clients (logs only)
3. **JSON-Safe Serialization:** `_error_payload` converts non-serializable objects to strings
4. **Fail-Fast Configuration:** Missing `OPENAI_API_KEY` raises `RuntimeError` at startup

## Scalability Considerations

### Current Limitations

- **Single Process:** Uvicorn runs one worker (development mode)
- **In-Memory Caching:** LRU caches don't share across processes
- **ChromaDB Embedded:** SQLite backend limits concurrent writes

### Scaling Strategies

#### Horizontal (Multiple Workers)

```bash
uvicorn src.api:app --workers 4
```

**Considerations:**
- Each worker loads its own ChromaDB connection (read-only safe)
- LRU caches duplicated per worker (minor memory overhead)
- Load balancer (nginx, Traefik) distributes requests

#### Vertical (Faster Hardware)

- More CPU cores → parallel embedding/generation
- SSD storage → faster ChromaDB queries
- More RAM → larger LRU caches

#### Database Upgrades

- **ChromaDB Client-Server:** Separate ChromaDB process, shared across workers
- **Pinecone/Weaviate:** Managed vector databases for massive scale
- **Pgvector:** PostgreSQL extension for vector search

#### Caching

- **Redis:** Cache frequent questions → answers
- **CDN:** Cache static documentation responses
- **Embedding Cache:** Store question embeddings (10-100x speedup for repeated queries)

## Security Considerations

### Current State

- **No Authentication:** Endpoints are open (suitable for internal services)
- **API Key Protection:** `OPENAI_API_KEY` loaded from environment, never logged
- **Input Validation:** Pydantic enforces length/type constraints

### Production Recommendations

1. **Add Authentication:**
   - API keys (custom header)
   - OAuth2 / JWT tokens
   - mTLS for service-to-service

2. **Rate Limiting:**
   - Per-IP or per-user limits
   - Use `slowapi` or external gateway (Kong, AWS API Gateway)

3. **Input Sanitization:**
   - Additional validation for prompt injection attempts
   - Content filtering for sensitive queries

4. **Secrets Management:**
   - AWS Secrets Manager / HashiCorp Vault
   - Never commit `.env` to version control

## Monitoring & Observability

### Current Logging

- **Level:** INFO (configurable via Python logging)
- **Format:** `%(asctime)s - %(levelname)s - %(message)s`
- **Output:** stdout (captured by Docker/Kubernetes)

### Recommended Additions

1. **Structured Logging:** JSON logs with request IDs, user IDs, trace context
2. **Metrics:**
   - Request latency (p50, p95, p99)
   - Error rates by type (4xx, 5xx, retries)
   - OpenAI API costs (tokens consumed)
   - Vector search performance
3. **Tracing:** OpenTelemetry for distributed tracing
4. **Alerting:** Slack/PagerDuty for high error rates, slow responses

## Future Enhancements

### Planned

- [ ] Token-by-token streaming for real-time generation feedback
- [ ] Hybrid search (vector + keyword BM25)
- [ ] Multi-document re-ranking with cross-encoders
- [ ] Conversation memory for follow-up questions
- [ ] Admin API for document management (upload, delete, re-index)

### Under Consideration

- [ ] Support for local LLMs (Ollama, llama.cpp)
- [ ] Multi-modal documents (images, tables)
- [ ] Query expansion for better retrieval recall
- [ ] Chunk relevance scoring (beyond cosine similarity)

---

## Key Takeaways

1. **RAG Pattern:** Combines retrieval (vector search) with generation (LLM) for grounded answers
2. **Resilience First:** Retry logic, error handling, and defensive coding throughout
3. **Simple but Extensible:** ChromaDB + FastAPI are easy to replace/upgrade as needs grow
4. **Observability Ready:** Logging and error tracking prepare for production monitoring
5. **Configuration-Driven:** All tunables exposed via environment variables
