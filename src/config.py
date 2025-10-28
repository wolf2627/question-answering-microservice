""" Takes care of configuration settings for the application. """

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import os

from dotenv import load_dotenv

# Loads environment variables from the .env file
load_dotenv()

@dataclass(frozen=True)
class Settings:
    docs_path: Path = Path(os.getenv("DOCS_PATH", "./documents"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    embeddings_path: Path = Path(os.getenv("CHROMA_PATH", "embeddings"))
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "30.0"))
    max_embed_retries: int = int(os.getenv("MAX_EMBED_RETRIES", "5"))

    stream_idle_timeout: float = float(os.getenv("STREAM_IDLE_TIMEOUT", "60.0"))
    stream_max_duration: float = float(os.getenv("STREAM_MAX_DURATION", "600.0"))

# Caches and returns the settings instance
# This ensures that settings are only loaded once.
@lru_cache()
def get_settings() -> Settings:
    return Settings()


