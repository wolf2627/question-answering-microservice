# Examples

Practical examples of using the Question Answering Microservice with various query types and formats.

## Prerequisites

Ensure the service is running:
```bash
make dev
```

And you've built the document index:
```bash
make index
```

---
## Basic Question Answering

### Simple Factual Question

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "<Your Question Here>"
  }'
```

**Response:**
```json
{
  "answer": "Answer for the Question",
  "sources": [
    {
      "chunk_id": "examplefile1__chunk_0",
      "document": "examplefile1",
      "score": 0.92
    },
    {
      "chunk_id": "examplefile2__chunk_1",
      "document": "examplefile2",
      "score": 0.85
    }
  ]
}
```

---

### Questions
1. Explain anotomy of eye
2. Who protects eye from dust and light?
3. Explain the function of Cornea
4. List devices that helps to improve hearing
5. What are surgical aids?
6. how image if formed on eyes?

```

---

### How-To Question

Note: These are sample question and answer for understanding

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I set up the project?"
  }'
```

**Response:**
```json
{
  "answer": "To set up the project: 1) Install uv, 2) Run 'uv sync' to install dependencies, 3) Create a .env file with your OPENAI_API_KEY, 4) Run 'make index' to build the document index, and 5) Run 'make dev' to start the server (citing docs__getting-started.md__chunk_2, docs__getting-started.md__chunk_3).",
  "sources": [
    {
      "chunk_id": "docs__getting-started.md__chunk_2",
      "document": "docs/getting-started.md",
      "score": 0.91
    },
    {
      "chunk_id": "docs__getting-started.md__chunk_3",
      "document": "docs/getting-started.md",
      "score": 0.87
    }
  ]
}
```

---

## Streaming Responses

### Enable Streaming

**Request:**
```bash
curl -N -X POST "http://localhost:8000/ask?stream=true" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main features?"
  }'
```

**Response Stream:**
```json
{"event": "context", "data": {"question": "What are the main features?", "sources": [{"chunk_id": "README.md__chunk_1", "document": "README.md", "score": 0.89}]}}
{"event": "answer", "data": {"answer": "The main features include semantic search, source attribution, streaming support, robust error handling, and production-ready design (citing README.md__chunk_1).", "sources": [{"chunk_id": "README.md__chunk_1", "document": "README.md", "score": 0.89}]}}
```

### Processing Streaming Events

**Python Example:**
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/ask?stream=true",
    json={"question": "What are the main features?"},
    stream=True
)

for line in response.iter_lines():
    if line:
        event = json.loads(line)
        if event["event"] == "context":
            print("Sources retrieved:", len(event["data"]["sources"]))
        elif event["event"] == "answer":
            print("Answer:", event["data"]["answer"])
        elif event["event"] == "error":
            print("Error:", event["data"]["error"]["message"])
```

**JavaScript Example:**
```javascript
fetch('http://localhost:8000/ask?stream=true', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'What are the main features?' })
})
.then(response => response.body)
.then(body => {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  
  function read() {
    reader.read().then(({ done, value }) => {
      if (done) return;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());
      
      lines.forEach(line => {
        const event = JSON.parse(line);
        if (event.event === 'answer') {
          console.log('Answer:', event.data.answer);
        }
      });
      
      read();
    });
  }
  
  read();
});
```

---

## Error Handling Examples

### Validation Error (Question Too Short)

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Hi"}'
```

**Response (422):**
```json
{
  "error": {
    "status": 422,
    "message": "Validation failed",
    "details": [
      {
        "loc": ["body", "question"],
        "msg": "ensure this value has at least 3 characters",
        "type": "value_error.any_str.min_length",
        "ctx": {"limit_value": 3}
      }
    ]
  }
}
```

---

### Missing Question Field

**Request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Response (422):**
```json
{
  "error": {
    "status": 422,
    "message": "Validation failed",
    "details": [
      {
        "loc": ["body", "question"],
        "msg": "field required",
        "type": "value_error.missing"
      }
    ]
  }
}
```

---

### Invalid API Key (502 Error)

**Request:**
```bash
# Assuming OPENAI_API_KEY is invalid/missing
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this?"}'
```

**Response (502):**
```json
{
  "error": {
    "status": 502,
    "message": "Context retrieval failed after retries"
  }
}
```

---