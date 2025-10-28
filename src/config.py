""" Takes care of configuration settings for the application. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
import os

from dotenv import load_dotenv

DEFAULT_RESPONSE_INSTRUCTIONS = (
    "You are a helpful AI assistant. Provide concise and accurate answers based on the provided context."
)

def _load_response_instructions() -> str:
    """Load response instructions from file, falling back to default when missing or empty."""
    path = Path(os.getenv("RESPONSE_INSTRUCTIONS_PATH", "instruction.txt"))
    try:
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    except FileNotFoundError:
        pass
    return DEFAULT_RESPONSE_INSTRUCTIONS

# Loads environment variables from the .env file
load_dotenv()

@dataclass(frozen=True)
class Settings:
    docs_path: Path = Path(os.getenv("DOCS_PATH", "./documents"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "30"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "50"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "5"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    embeddings_path: Path = Path(os.getenv("CHROMA_PATH", "embeddings"))
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "30.0"))
    max_embed_retries: int = int(os.getenv("MAX_EMBED_RETRIES", "5"))

    stream_idle_timeout: float = float(os.getenv("STREAM_IDLE_TIMEOUT", "60.0"))
    stream_max_duration: float = float(os.getenv("STREAM_MAX_DURATION", "600.0"))

    top_k: int = int(os.getenv("TOP_K", "10"))

    response_instructions_path: Path = Path(os.getenv("RESPONSE_INSTRUCTIONS_PATH", "instruction.txt"))
    response_instructions: str = field(default_factory=_load_response_instructions)
    response_model: str = os.getenv("RESPONSE_MODEL", "gpt-4.1-nano")
    response_max_tokens= int = int(os.getenv("RESPONSE_MAX_TOKENS", "300"))
    response_temperature: float = float(os.getenv("RESPONSE_TEMPERATURE", "0.0"))

# Caches and returns the settings instance
# This ensures that settings are only loaded once.
@lru_cache()
def get_settings() -> Settings:
    return Settings()

