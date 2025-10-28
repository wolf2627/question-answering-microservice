""" FastAPI application exposing the question-answering microservice. """

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from functools import lru_cache
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, RetryError

from src.config import Settings, get_settings
from src.openai_client import OpenAIClient, OpenAIClientConfig
from src.vector_store import get_vector_store, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Question Answering Service", version="1.0.0")

# Models / validation
class SourceAttribution(BaseModel):
    chunk_id: str
    document: str
    score: float

class AskRequest(BaseModel):
    """Request payload for /ask (question must be non-empty and reasonably sized)."""
    question: str = Field(..., min_length=3, max_length=500, description="Natural language question to answer")

    @validator("question", pre=True)
    def strip_and_validate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("question must be a string")
        q = v.strip()
        if not q:
            raise ValueError("question must not be empty or whitespace")
        if len(q) < 3:
            raise ValueError("question too short after trimming")
        return q

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceAttribution]

# Cached dependency factories
@lru_cache(maxsize=1)
def _get_vector_store_cached(settings: Settings) -> VectorStore:
    return get_vector_store(settings)

@lru_cache(maxsize=1)
def _get_openai_client_cached(settings: Settings) -> OpenAIClient:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    config = OpenAIClientConfig(
        api_key=settings.openai_api_key,
        embed_model=settings.embed_model,
        timeout=settings.openai_timeout,
        max_retries=settings.max_embed_retries,
    )
    return OpenAIClient(config)

# Retry/backoff configuration
_MAX_RETRIES = 5
_WAIT = wait_exponential(multiplier=1, min=1, max=30)  # exponential backoff with cap

def _should_retry(exc: Exception) -> bool:
    """Return True if exception appears transient (429 or 5xx-like or network timeouts)."""
    # If the exception has an HTTP-like status attribute
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        if status is not None:
            code = int(status)
            if code == 429 or 500 <= code < 600:
                return True
    except Exception:
        # ignore conversion errors and fall back to message-based checks
        pass

    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return True
    if "server error" in msg or "internal" in msg or "temporar" in msg:
        return True

    # Common transient network errors - treat as retryable
    if isinstance(exc, TimeoutError):
        return True

    return False

# ---------------------------
# Exception handlers (centralized & sanitized responses)
# ---------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Keep the detailed list shape (422) — useful for client debugging, but can be shortened
    logger.info("Validation error for %s %s: %s", request.method, request.url, exc.errors())
    return _error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Validation failed",
        details=exc.errors(),
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Normalize detail shape while preserving status code.
    detail = exc.detail
    if isinstance(detail, str) and detail.strip():
        message = detail.strip()
        details = None
    else:
        message = "Request failed"
        details = detail
    logger.info(
        "HTTP exception %s for %s %s: %s",
        exc.status_code,
        request.method,
        request.url,
        {"message": message, "details": details},
    )
    return _error_response(status_code=exc.status_code, message=message, details=details)

@app.exception_handler(RetryError)
async def retry_exception_handler(request: Request, exc: RetryError):
    # Tenacity exhausted retries — return a sanitized 502 while logging the inner exception
    logger.error("Retries exhausted for request %s %s: %s", request.method, request.url, traceback.format_exc())
    return _error_response(
        status_code=status.HTTP_502_BAD_GATEWAY,
        message="Upstream service temporarily unavailable; please retry later.",
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Catch-all - do NOT expose exc to the client. Log full traceback server-side.
    logger.exception("Unhandled exception handling request %s %s: %s", request.method, request.url, traceback.format_exc())
    return _error_response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Internal server error")

# Synchronous wrappers with tenacity (called from threads)
@retry(
    reraise=True,
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=_WAIT,
    retry=retry_if_exception(_should_retry),
)
def _retrieve_context_sync(vector_store: VectorStore, question: str, client: OpenAIClient, top_k: int):
    """Synchronous retrieval call wrapped with retries. Intended to be run in a thread."""
    return vector_store.similarity_search_text(question, client=client, top_k=top_k)


@retry(
    reraise=True,
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=_WAIT,
    retry=retry_if_exception(_should_retry),
)
def _generate_answer_sync(openai_client: OpenAIClient, instructions: str, prompt: str, model: str, max_output_tokens: int, temperature: float):
    """Synchronous generation call wrapped with retries. Intended to be run in a thread."""
    return openai_client.generate_answer(
        instructions=instructions,
        prompt=prompt,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


# Async helpers calling the sync wrappers via threads
async def _retrieve_context(*, question: str, openai_client: OpenAIClient, vector_store: VectorStore, top_k: int) -> list[RetrievedChunk]:
    """Retrieve relevant document chunks for the question (with retries for transient failures)."""
    try:
        chunks = await asyncio.to_thread(_retrieve_context_sync, vector_store, question, openai_client, top_k)
        return chunks
    except RetryError as re:
        logger.error("Retries exhausted during retrieval: %s", re)
        # Do not expose internal exception text to the client
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Context retrieval failed after retries")
    except Exception as e:
        logger.exception("Error during context retrieval")
        # Sanitize for client
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Context retrieval failed") from e


async def _generate_answer(*, question: str, chunks: list[RetrievedChunk], openai_client: OpenAIClient, settings: Settings) -> str:
    """Generate an answer given the question and retrieved chunks (with retries for transient failures)."""
    if not chunks:
        context_prompt = "No context passages were retrieved. Answer conservatively."
    else:
        formatted_chunks = []
        for chunk in chunks:
            document = str(
                chunk.metadata.get("source_path")
                or chunk.metadata.get("document_id")
                or chunk.chunk_id,
            )
            formatted_chunks.append(
                f"{chunk.chunk_id} ({document}):\n{chunk.content.strip()}"
            )
        context_prompt = "\n\n".join(formatted_chunks)

    prompt = (
        "Context passages:\n"
        f"{context_prompt}\n\n"
        f"Question: {question}\n"
        "Respond with a factual answer that cites chunk identifiers in parentheses."
    )

    try:
        answer_text = await asyncio.to_thread(
            _generate_answer_sync,
            openai_client,
            settings.response_instructions,
            prompt,
            settings.response_model,
            settings.response_max_tokens,
            settings.response_temperature,
        )
        if not answer_text:
            raise RuntimeError("Empty response from OpenAI")
        return answer_text.strip()
    except RetryError as re:
        logger.error("Retries exhausted during generation: %s", re)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation failed after retries")
    except HTTPException:
        # propagate HTTPException unchanged
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Generation failed")
        # Sanitize for client
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation failed") from exc

# Helpers & deps used by endpoints
def _sources_from_chunks(chunks: list[RetrievedChunk]) -> list[SourceAttribution]:
    sources: list[SourceAttribution] = []
    for chunk in chunks:
        document = str(
            chunk.metadata.get("source_path")
            or chunk.metadata.get("document_id")
            or chunk.chunk_id,
        )
        sources.append(
            SourceAttribution(
                chunk_id=chunk.chunk_id,
                document=document,
                score=float(chunk.score),
            )
        )
    return sources

def get_settings_dep() -> Settings:
    return get_settings()

def get_vector_store_dep(settings: Settings = Depends(get_settings_dep)) -> VectorStore:
    return _get_vector_store_cached(settings)

def get_openai_client_dep(settings: Settings = Depends(get_settings_dep)) -> OpenAIClient:
    return _get_openai_client_cached(settings)

def _error_payload(status_code: int, message: str, *, details: object | None = None) -> dict[str, object]:
    def _json_safe(value: object) -> object:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, (set, tuple)):
            return [_json_safe(item) for item in value]
        if isinstance(value, dict):
            return {key: _json_safe(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    error: dict[str, object] = {
        "status": int(status_code),
        "message": str(message),
    }
    if details is not None:
        error["details"] = _json_safe(details)
    return {"error": error}


def _error_response(status_code: int, message: str, *, details: object | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(status_code, message, details=details),
    )


def _serialize_event(event_name: str, payload: dict) -> bytes:
    return (json.dumps({"event": event_name, "data": payload}) + "\n").encode("utf-8")

# Endpoints 
@app.post("/ask", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    stream: bool = Query(False, description="Stream chunked progress events"),
    settings: Settings = Depends(get_settings_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
    openai_client: OpenAIClient = Depends(get_openai_client_dep),
):
    logger.info("Received question request: %s", body.question)

    # Retrieve context with retries handled in the wrapper
    chunks = await _retrieve_context(
        question=body.question,
        vector_store=vector_store,
        openai_client=openai_client,
        top_k=settings.top_k,
    )

    if not stream:
        answer_text = await _generate_answer(
            question=body.question,
            chunks=chunks,
            openai_client=openai_client,
            settings=settings,
        )
        response = AskResponse(answer=answer_text, sources=_sources_from_chunks(chunks))
        return JSONResponse(status_code=status.HTTP_200_OK, content=response.model_dump())

    async def event_stream() -> AsyncIterator[bytes]:
        sources = _sources_from_chunks(chunks)
        yield _serialize_event(
            "context",
            {
                "question": body.question,
                "sources": [source.model_dump() for source in sources],
            },
        )

        try:
            answer_text = await _generate_answer(
                question=body.question,
                chunks=chunks,
                openai_client=openai_client,
                settings=settings,
            )
            yield _serialize_event(
                "answer",
                {
                    "answer": answer_text,
                    "sources": [source.model_dump() for source in sources],
                },
            )
        except HTTPException as exc:
            # exc.detail may be structured (dict/list) thanks to our handlers
            status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
            detail_payload = exc.detail

            if (
                isinstance(detail_payload, dict)
                and isinstance(detail_payload.get("error"), dict)
                and detail_payload["error"].get("message")
            ):
                error_block = detail_payload["error"]
                message = str(error_block.get("message", "Request failed"))
                details = error_block.get("details")
            elif isinstance(detail_payload, str) and detail_payload.strip():
                message = detail_payload.strip()
                details = None
            else:
                message = "Request failed"
                details = detail_payload

            yield _serialize_event(
                "error",
                _error_payload(
                    status_code,
                    message,
                    details=details,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Streaming error")
            yield _serialize_event(
                "error",
                _error_payload(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "Internal streaming error",
                ),
            )

    return StreamingResponse(event_stream(), media_type="application/json")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_200_OK)

# manual test endpoint for general info
@app.get("/", response_class=JSONResponse)
async def root() -> JSONResponse:
    """Simple root endpoint with service info."""
    info = {
        "service": "Question Answering Service",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST endpoint to ask a question",
            "/health": "GET health check endpoint",
        },
    }
    return JSONResponse(content=info, status_code=status.HTTP_200_OK)