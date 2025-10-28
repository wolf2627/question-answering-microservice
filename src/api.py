""" FastAPI application exposing the question-answering microservice. """

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.openai_client import OpenAIClient, OpenAIClientConfig
from src.vector_store import get_vector_store, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Question Answering Service", version="1.0.0")

class SourceAttribution(BaseModel):
    chunk_id: str
    document: str
    score: float

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question to answer")

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceAttribution]

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

async def _retrieve_context(*, question:str,openai_client=OpenAIClient ,vector_store: VectorStore, top_k: int) -> list[RetrievedChunk]:
    """Retrieve relevant document chunks for the question."""
    try:
        chunks = await asyncio.to_thread(vector_store.similarity_search_text, question, client=openai_client, top_k = top_k)
        return chunks
    except Exception as e:
        logger.error("Error during context retrieval: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)) from e

async def _generate_answer( *, question: str, chunks: list[RetrievedChunk], openai_client: OpenAIClient, settings: Settings) -> str:
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
            openai_client.generate_answer,
            instructions=settings.response_instructions,
            prompt=prompt,
            model=settings.response_model,
            max_output_tokens=settings.response_max_tokens,
            temperature=settings.response_temperature,
        )
        if not answer_text:
            raise RuntimeError("Empty response from OpenAI")
        return answer_text.strip()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Generation failed")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

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


def _serialize_event(event_name: str, payload: dict) -> bytes:
    return (json.dumps({"event": event_name, "data": payload}) + "\n").encode("utf-8")


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    stream: bool = Query(False, description="Stream chunked progress events"),
    settings: Settings = Depends(get_settings_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
    openai_client: OpenAIClient = Depends(get_openai_client_dep),
):
    logger.info("Received question request: %s", body.question)

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
            yield _serialize_event("error", {"detail": exc.detail})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Streaming error")
            yield _serialize_event("error", {"detail": str(exc)})

    return StreamingResponse(event_stream(), media_type="application/json")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_200_OK)