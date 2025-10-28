# QA Orchestration — Workflow Setup Guide

### Overview
This n8n workflow automates **Question Answering (QA)** — it receives a question, queries the backend API for an answer, fetches supporting documents, measures response time, logs results, and returns a formatted response to the client.  

It also includes error handling and email alerts.

---

## Setup Steps

### 1. Import the Workflow
- Open your **n8n** instance.
- Go to **Workflows → Import from File**.
- Upload: `qa_orchestration.json`.

### 2. Configure Nodes
- `Fetch Answer` node with microservice api url. `\ask` is the endpoint.
- `Log` node with mongodb crendentials and add proper collection.
- `Failure Alert` node with gmail credentials. also set the `To` mail id.


### 2. Test Webhook
- Note the generated **Webhook URL** (e.g., `/webhook/ask`).
- Test the endpoint using Postman:
  ```bash
  POST https://<your-n8n-domain>/webhook/ask
  Content-Type: application/json

  {
    "question": "Explain anatomy of eyes"
  }
