# API Reference

Complete documentation for all endpoints exposed by the Question Answering Microservice.

## Base URL

```
http://localhost:8000
```

## Endpoints

### POST `/ask`

Ask a natural language question and receive an answer with source citations.

#### Request

**Headers:**
```
Content-Type: application/json
```

**Body Parameters:**

| Parameter | Type | Required | Constraints | Description |
|-----------|------|----------|-------------|-------------|
| `question` | string | Yes | 3-500 chars | Natural language question |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stream` | boolean | `false` | Enable streaming response |

**Example Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?"}'
```

#### Response (Standard Mode)

**Status Code:** `200 OK`

**Body Schema:**
```json
{
  "answer": "string",
  "sources": [
    {
      "chunk_id": "string",
      "document": "string",
      "score": 0.0
    }
  ]
}
```

**Fields:**

- `answer` (string): Generated answer citing chunk identifiers in parentheses
- `sources` (array): List of document chunks used for context
  - `chunk_id` (string): Unique identifier for the chunk
  - `document` (string): Source document path
  - `score` (float): Similarity score (0-1, higher is better)

**Example Response:**
```json
{
  "answer": "The key features include semantic search, source attribution, and streaming support (citing docs__readme.md__chunk_2).",
  "sources": [
    {
      "chunk_id": "docs__readme.md__chunk_2",
      "document": "docs/readme.md",
      "score": 0.89
    },
    {
      "chunk_id": "docs__readme.md__chunk_3",
      "document": "docs/readme.md",
      "score": 0.76
    }
  ]
}
```

#### Response (Streaming Mode)

**Status Code:** `200 OK`

**Content-Type:** `application/json` (newline-delimited)

**Event Stream Format:**

Each line is a JSON object with `event` and `data` fields:

```json
{"event": "context", "data": {"question": "...", "sources": [...]}}
{"event": "answer", "data": {"answer": "...", "sources": [...]}}
```

**Event Types:**

1. **context** - Sent immediately with retrieved sources
   ```json
   {
     "event": "context",
     "data": {
       "question": "What is the main topic of the documents?",
       "sources": [
         {
           "chunk_id": "docs__readme.md__chunk_2",
           "document": "docs/readme.md",
           "score": 0.89
         }
       ]
     }
   }
   ```

2. **answer** - Sent after generation completes
   ```json
   {
     "event": "answer",
     "data": {
       "answer": "The main topics include...",
       "sources": [...]
     }
   }
   ```

3. **error** - Sent if generation fails
   ```json
   {
     "event": "error",
     "data": {
       "error": {
         "status": 502,
         "message": "Generation failed after retries"
       }
     }
   }
   ```

**Example Streaming Request:**
```bash
curl -X POST "http://localhost:8000/ask?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key features?"}'
```

---

### GET `/health`

Health check endpoint to verify the service is running.

#### Request

```bash
curl http://localhost:8000/health
```

#### Response

**Status Code:** `200 OK`

**Body:**
```json
{
  "status": "ok"
}
```

---

## Error Responses

All errors follow a unified structure:

### Error Schema

```json
{
  "error": {
    "status": 422,
    "message": "string",
    "details": {}
  }
}
```

**Fields:**

- `error.status` (integer): HTTP status code
- `error.message` (string): Human-readable error description
- `error.details` (object, optional): Additional context (e.g., validation errors)

### Common Error Status Codes

#### 422 Unprocessable Entity

**Validation Error** - Request body failed validation

**Example:**
```json
{
  "error": {
    "status": 422,
    "message": "Validation failed",
    "details": [
      {
        "loc": ["body", "question"],
        "msg": "ensure this value has at least 3 characters",
        "type": "value_error.any_str.min_length"
      }
    ]
  }
}
```

**Common Causes:**
- Question is too short (<3 chars)
- Question is too long (>2000 chars)
- Question field is missing or not a string

#### 502 Bad Gateway

**Upstream Service Error** - OpenAI API or ChromaDB failure

**Example:**
```json
{
  "error": {
    "status": 502,
    "message": "Context retrieval failed after retries"
  }
}
```

**Common Causes:**
- OpenAI API rate limits exceeded
- Network timeout
- Invalid API key
- ChromaDB not accessible

#### 500 Internal Server Error

**Unexpected Error** - Unhandled server exception

**Example:**
```json
{
  "error": {
    "status": 500,
    "message": "Internal server error"
  }
}
```

---

## Retry Behavior

The service implements automatic retry with exponential backoff for transient failures:

- **Max Retries:** 5 attempts
- **Backoff Strategy:** Exponential (1s, 2s, 4s, 8s, 16s) with jitter
- **Retryable Conditions:**
  - HTTP 429 (Rate Limit)
  - HTTP 5xx (Server Errors)
  - Network timeouts
  - Transient OpenAI API errors

Non-retryable errors (e.g., validation failures, bad API keys) fail immediately.

---

## Rate Limits

Rate limits are imposed by the OpenAI API:

- **Embeddings:** Varies by tier (see [OpenAI documentation](https://platform.openai.com/docs/guides/rate-limits))
- **Completions:** Varies by model and tier

The service automatically handles rate limits with retry/backoff. If limits are persistently exceeded, you'll receive a `502` error after exhausting retries.

---

## Interactive Documentation

FastAPI provides auto-generated interactive documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

Both interfaces allow you to:
- Explore all endpoints
- View request/response schemas
- Test API calls directly from the browser
