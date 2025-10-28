""" OpenAI client wrapper with retry/backoff helpers."""

from __future__ import annotations
import asyncio
import random
import time
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import TypeVar, Optional, Union, AsyncGenerator

from openai import OpenAI
from openai import APIError, RateLimitError, APIStatusError  # Try and modify if needed.

T = TypeVar("T") # Generic type variable

@dataclass(frozen=True)
class OpenAIClientConfig:
    api_key: str
    embed_model: str
    timeout: float = 30.0
    max_retries: int = 5
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    # new streaming timeout parameters:
    stream_idle_timeout: Optional[float] = None  # e.g., seconds of no chunks before abort
    stream_max_duration: Optional[float] = None  # e.g., max total streaming seconds

class OpenAIClient:
    """Lightweight client with retry/backoff helpers for OpenAI operations."""
    def __init__(self, config: OpenAIClientConfig) -> None:
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        self._config = config
        self._client = OpenAI(api_key=config.api_key, timeout=config.timeout)

    def embed_texts(self, texts: Sequence[str], *, model: Optional[str] = None) -> list[list[float]]:
        """Embed multiple texts"""
        if not texts:
            return []
        target_model = model or self._config.embed_model

        def operation() -> list[list[float]]:
            response = self._client.embeddings.create(model=target_model, input=list(texts), dimensions=1536)
            return [item.embedding for item in response.data]

        return self._execute_with_retry(operation)

    def embed_text(self, text: str, *, model: Optional[str] = None) -> list[float]:
        """Embed a single text"""

        embeddings = self.embed_texts([text], model=model)
        return embeddings[0]

    def generate_answer(self, *, instructions: str,prompt: str, model: str, max_output_tokens: int, temperature: Optional[float] = None  ) -> str:
        """Synchronous generation (non-streaming)."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        def operation() -> str:
            # Add Temperature if provided
            response = self._client.responses.create( model=model, instructions=instructions, input=prompt, max_output_tokens=max_output_tokens)
            return getattr(response, "output_text", "").strip()

        return self._execute_with_retry(operation)

    async def generate_answer_stream(self, *, instructions: str, prompt: str, model: str, max_output_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
        """Asynchronous streaming generation — yields chunks as they arrive."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        async def operation_stream() -> AsyncGenerator[str, None]:
            resp = await self._client.responses.create(
                model=model,
                instructions=instructions,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                stream=True
            )
            async for event in resp:
                # Extract chunk text from event structure
                # Example: event.delta.text or event.output_text
                delta = getattr(event, "delta", None)
                if delta and hasattr(delta, "text"):
                    yield delta.text
                elif getattr(event, "output_text", None):
                    yield event.output_text
               # TODO: Handle Event types if needed.

        return self._stream_with_retry(
            operation_stream,
            max_duration=self._config.stream_max_duration,
            idle_timeout=self._config.stream_idle_timeout
        )

    def _execute_with_retry(self, operation: Callable[[], T]) -> T:
        """Executes a synchronous operation with retry and exponential backoff."""
        attempt = 0
        backoff = self._config.initial_backoff
        while True:
            try:
                return operation()
            except (APIError, RateLimitError, APIStatusError) as exc:
                attempt += 1
                if attempt > self._config.max_retries:
                    raise
                jitter = random.uniform(0.0, 0.5)
                sleep_for = min(backoff, self._config.max_backoff) + jitter
                time.sleep(sleep_for)
                backoff *= 2

    async def _stream_with_retry( self, operation: Callable[[], AsyncGenerator[str, None]], *, max_duration: Optional[float] = None, idle_timeout: Optional[float] = None ) -> AsyncGenerator[str, None]:

        """ Executes an asynchronous streaming operation with retry and exponential backoff."""
        attempt = 0
        backoff = self._config.initial_backoff

        while True:
            attempt += 1
            start_time = time.monotonic()
            last_chunk_time = start_time
            try:
                async for chunk in await operation():
                    now = time.monotonic()
                    last_chunk_time = now
                    yield chunk

                    # Check total duration
                    if max_duration is not None and (now - start_time) > max_duration:
                        raise RuntimeError(f"Stream exceeded max duration of {max_duration}s")

                # Completed the stream normally
                return

            except (APIError, RateLimitError, APIStatusError, RuntimeError) as exc:
                now = time.monotonic()
                # Check idle timeout
                if idle_timeout is not None and (now - last_chunk_time) > idle_timeout:
                    exc = RuntimeError(f"No stream chunk received for {(now - last_chunk_time):.1f}s (> {idle_timeout}s)")

                if attempt > self._config.max_retries:
                    raise exc

                jitter = random.uniform(0.0, 0.5)
                sleep_for = min(backoff, self._config.max_backoff) + jitter
                await asyncio.sleep(sleep_for)
                backoff *= 2
                # then retry loop continues

