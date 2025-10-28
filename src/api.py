""" FastAPI application exposing the question-answering microservice. """

from __future__ import annotations

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI(title="Question Answering Service", version="1.0.0")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_200_OK)